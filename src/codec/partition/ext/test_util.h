// Shared test helpers for the extended-partition component.

#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "codec/partition/tree/partition.h"
#include "codec/partition/ext/partition_ext.h"

namespace pe = av2enc::partition_ext;
namespace p = av2enc::partition;

extern int g_failures;

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      ++g_failures;                                                          \
      std::cerr << __FILE__ << ':' << __LINE__                               \
                << ": CHECK failed: " #condition << '\n';                    \
    }                                                                        \
  } while (false)

#define CHECK_EQ(actual, expected)                                           \
  do {                                                                       \
    const long long actual_value = static_cast<long long>(actual);           \
    const long long expected_value = static_cast<long long>(expected);       \
    if (actual_value != expected_value) {                                    \
      ++g_failures;                                                          \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK_EQ failed: "      \
                << #actual << " = " << actual_value << ", expected "         \
                << expected_value << '\n';                                   \
    }                                                                        \
  } while (false)

#define CHECK_STR(actual, expected)                                          \
  do {                                                                       \
    const std::string actual_value = (actual);                               \
    const std::string expected_value = (expected);                           \
    if (actual_value != expected_value) {                                    \
      ++g_failures;                                                          \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK_STR failed: "     \
                << #actual << " = \"" << actual_value << "\", expected \""   \
                << expected_value << "\"\n";                                 \
    }                                                                        \
  } while (false)

inline p::FrameGeometry geometry_420(int width, int height, int sb_size) {
  p::FrameGeometry geometry;
  geometry.width = width;
  geometry.height = height;
  geometry.sb_size = sb_size;
  return geometry;
}

inline pe::Tools ext_tools(bool ext = true, bool uneven = true) {
  pe::Tools tools;
  tools.enable_ext_partitions = ext;
  tools.enable_uneven_4way_partitions = uneven;
  return tools;
}

inline std::string base_signature(const p::PartitionNode& node) {
  std::string out = p::block_size_name(node.size);
  out += '@';
  out += std::to_string(node.mi_row);
  out += ',';
  out += std::to_string(node.mi_col);
  out += ':';
  out += p::partition_name(node.partition);
  out += ':';
  out += node.forced ? 'F' : 'S';
  out += '(';
  for (const p::PartitionNode& child : node.children) {
    out += base_signature(child);
  }
  out += ')';
  return out;
}

// A cost model with exactly one finite partition candidate per node.  The leaf
// cost is area^2, so any split whose children stop splitting is cheaper than
// NONE.  This makes the chosen tree follow `chooser` exactly wherever the
// spec allows the requested partition.
inline pe::CostFunction choose_partition(
    std::function<pe::PartitionType(const pe::NodeInfo&)> chooser) {
  return [chooser = std::move(chooser)](const pe::NodeInfo& info) {
    const double w = static_cast<double>(p::block_width(info.size));
    const double h = static_cast<double>(p::block_height(info.size));
    const double area = w * h;
    pe::NodeCosts costs;
    costs.leaf = area * area;
    costs.none_signal = pe::kInfiniteCost;
    costs.split_signal = pe::kInfiniteCost;
    costs.horz_signal = pe::kInfiniteCost;
    costs.vert_signal = pe::kInfiniteCost;
    costs.horz3_signal = pe::kInfiniteCost;
    costs.vert3_signal = pe::kInfiniteCost;
    costs.horz4a_signal = pe::kInfiniteCost;
    costs.horz4b_signal = pe::kInfiniteCost;
    costs.vert4a_signal = pe::kInfiniteCost;
    costs.vert4b_signal = pe::kInfiniteCost;
    pe::PartitionType want = chooser(info);
    if (!info.allowed.is_allowed(want)) {
      // The spec always forces at least one partition.  Keep the model total:
      // prefer the boundary-implied partition, then the first allowed one.
      if (info.boundary.implied &&
          info.allowed.is_allowed(info.boundary.implied_partition)) {
        want = info.boundary.implied_partition;
      } else {
        for (int i = 0; i < pe::kPartitionCount; ++i) {
          const pe::PartitionType candidate = static_cast<pe::PartitionType>(i);
          if (info.allowed.is_allowed(candidate)) {
            want = candidate;
            break;
          }
        }
      }
    }
    switch (want) {
      case pe::PartitionType::kNone:
        costs.none_signal = 0.0;
        break;
      case pe::PartitionType::kSplit:
        costs.split_signal = 0.0;
        break;
      case pe::PartitionType::kHorz:
        costs.horz_signal = 0.0;
        break;
      case pe::PartitionType::kVert:
        costs.vert_signal = 0.0;
        break;
      case pe::PartitionType::kHorz3:
        costs.horz3_signal = 0.0;
        break;
      case pe::PartitionType::kVert3:
        costs.vert3_signal = 0.0;
        break;
      case pe::PartitionType::kHorz4A:
        costs.horz4a_signal = 0.0;
        break;
      case pe::PartitionType::kHorz4B:
        costs.horz4b_signal = 0.0;
        break;
      case pe::PartitionType::kVert4A:
        costs.vert4a_signal = 0.0;
        break;
      case pe::PartitionType::kVert4B:
        costs.vert4b_signal = 0.0;
        break;
    }
    return costs;
  };
}

// A deterministic pseudo-random cost model: every partition gets a random
// signal cost, all finite.  Used for determinism and compatibility tests.
inline pe::CostFunction random_cost(unsigned seed) {
  return [seed](const pe::NodeInfo& info) {
    unsigned state = seed * 2654435761u;
    state ^= static_cast<unsigned>(info.mi_row) * 2246822519u;
    state ^= static_cast<unsigned>(info.mi_col) * 3266489917u;
    state ^= static_cast<unsigned>(info.size) * 668265263u;
    state ^= static_cast<unsigned>(info.depth) * 374761393u;
    auto next = [&state]() {
      state = state * 1664525u + 1013904223u;
      return static_cast<double>((state >> 8u) & 0xFFFFu) / 65536.0;
    };
    const double w = static_cast<double>(p::block_width(info.size));
    const double h = static_cast<double>(p::block_height(info.size));
    pe::NodeCosts costs;
    costs.leaf = (w * h) * (0.5 + next());
    costs.none_signal = next() * 20.0;
    costs.split_signal = next() * 20.0;
    costs.horz_signal = next() * 20.0;
    costs.vert_signal = next() * 20.0;
    costs.horz3_signal = next() * 20.0;
    costs.vert3_signal = next() * 20.0;
    costs.horz4a_signal = next() * 20.0;
    costs.horz4b_signal = next() * 20.0;
    costs.vert4a_signal = next() * 20.0;
    costs.vert4b_signal = next() * 20.0;
    return costs;
  };
}

inline std::string leaf_summary(const pe::PartitionNode& node) {
  std::vector<pe::LeafBlock> leaves;
  pe::collect_leaves(node, leaves);
  std::string out;
  for (const pe::LeafBlock& leaf : leaves) {
    out += p::block_size_name(leaf.size);
    out += '@';
    out += std::to_string(leaf.mi_row);
    out += ',';
    out += std::to_string(leaf.mi_col);
    out += ':';
    out += std::to_string(leaf.has_chroma ? 1 : 0);
    out += ':';
    out += p::block_size_name(leaf.chroma.size);
    out += '@';
    out += std::to_string(leaf.chroma.mi_row);
    out += ',';
    out += std::to_string(leaf.chroma.mi_col);
    out += ';';
  }
  return out;
}
