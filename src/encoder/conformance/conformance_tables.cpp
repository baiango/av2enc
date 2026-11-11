#include "encoder/conformance/conformance_tables.h"

#include <array>
#include <cstddef>

namespace av2enc::conformance {
namespace {

// ---------------------------------------------------------------------------
// Table A.1 / A.2 (p1101-1102)
// ---------------------------------------------------------------------------

constexpr int kChroma420 = 0;  // CHROMA_FORMAT_420
constexpr int kChroma400 = 1;  // CHROMA_FORMAT_400
constexpr int kChroma444 = 2;  // CHROMA_FORMAT_444
constexpr int kChroma422 = 3;  // CHROMA_FORMAT_422

constexpr std::array<ProfileInfo, 32> kProfiles = {{
    // 0: Main_420_10_IP0
    {0, "Main_420_10_IP0", "profile 0", 0, 0, 15, 1.0,
     {kChroma400, kChroma420, 0, 0}, 2, false, false},
    // 1: Main_420_10_IP1
    {1, "Main_420_10_IP1", "profile 1", 1, 0, 15, 1.0,
     {kChroma400, kChroma420, 0, 0}, 2, false, false},
    // 2: Main_420_10_IP2
    {2, "Main_420_10_IP2", "profile 2", 2, 0, 15, 1.0,
     {kChroma400, kChroma420, 0, 0}, 2, false, false},
    // 3: Main_422_10_IP1
    {3, "Main_422_10_IP1", "profile 3", 1, 1, 20, 1.667,
     {kChroma400, kChroma420, kChroma422, 0}, 3, false, false},
    // 4: Main_444_10_IP1
    {4, "Main_444_10_IP1", "profile 4", 1, 2, 30, 2.5,
     {kChroma400, kChroma420, kChroma444, 0}, 3, false, false},
    // 5..30: Reserved
    {5, "Reserved", "profile 5 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {6, "Reserved", "profile 6 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {7, "Reserved", "profile 7 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {8, "Reserved", "profile 8 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {9, "Reserved", "profile 9 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {10, "Reserved", "profile 10 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {11, "Reserved", "profile 11 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {12, "Reserved", "profile 12 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {13, "Reserved", "profile 13 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {14, "Reserved", "profile 14 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {15, "Reserved", "profile 15 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {16, "Reserved", "profile 16 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {17, "Reserved", "profile 17 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {18, "Reserved", "profile 18 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {19, "Reserved", "profile 19 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {20, "Reserved", "profile 20 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {21, "Reserved", "profile 21 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {22, "Reserved", "profile 22 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {23, "Reserved", "profile 23 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {24, "Reserved", "profile 24 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {25, "Reserved", "profile 25 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {26, "Reserved", "profile 26 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {27, "Reserved", "profile 27 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {28, "Reserved", "profile 28 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {29, "Reserved", "profile 29 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    {30, "Reserved", "profile 30 (reserved)", -1, 0, 0, 0.0, {0, 0, 0, 0}, 0,
     true, false},
    // 31: Configurable — constraints come from chroma_format_idc,
    // bit_depth_idc and SeqMaxMlayerCnt (p1102); no Table A.2 row.
    {31, "Configurable", "profile 31 (Configurable)", -1, 0, 0, 0.0,
     {kChroma400, kChroma420, kChroma422, kChroma444}, 4, false, true},
}};

// ---------------------------------------------------------------------------
// Tables A.7/A.8/A.9 (p1104-1106)
// ---------------------------------------------------------------------------

constexpr std::array<LevelInfo, 22> kLevels = {{
    // idx, name, MaxPicSize, MaxHSize, MaxVSize, MaxDisplayRate,
    // MaxDecodeRate, MaxHeaderRate, MainMbps, HighMbps, MainCR, HighCR,
    // MaxTiles, MaxTileCols, example
    {0, "2.0", 147456, 640, 640, 4423680, 5529600, 150, 1.5, 0.0, 2, 0, 8, 4,
     "426x240@30fps"},
    {1, "2.1", 278784, 880, 880, 8363520, 10454400, 150, 3.0, 0.0, 2, 0, 8, 4,
     "640x360@30fps"},
    {2, "3.0", 665856, 1360, 1360, 19975680, 24969600, 150, 6.0, 0.0, 2, 0,
     16, 6, "854x480@30fps"},
    {3, "3.1", 1065024, 1720, 1720, 31950720, 39938400, 150, 10.0, 0.0, 2, 0,
     16, 6, "1280x720@30fps"},
    {4, "4.0", 2359296, 2560, 2560, 70778880, 77856768, 300, 12.0, 30.0, 4, 4,
     32, 8, "1920x1080@30fps"},
    {5, "4.1", 2359296, 2560, 2560, 141557760, 155713536, 300, 20.0, 50.0, 4,
     4, 32, 8, "1920x1080@60fps"},
    {6, "5.0", 8912896, 4975, 4975, 267386880, 273715200, 300, 30.0, 100.0, 6,
     4, 64, 8, "3840x2160@30fps"},
    {7, "5.1", 8912896, 4975, 4975, 534773760, 547430400, 300, 40.0, 160.0, 8,
     4, 64, 8, "3840x2160@60fps"},
    {8, "5.2", 8912896, 4975, 4975, 1069547520, 1094860800, 300, 60.0, 240.0,
     8, 4, 64, 8, "3840x2160@120fps"},
    {9, "5.3", 8912896, 4975, 4975, 1069547520, 1176502272, 300, 60.0, 240.0,
     8, 4, 64, 8, "3840x2160@120fps"},
    {10, "6.0", 35651584, 9951, 9951, 1069547520, 1176502272, 300, 60.0, 240.0,
     8, 4, 128, 16, "7680x4320@30fps"},
    {11, "6.1", 35651584, 9951, 9951, 2139095040, 2189721600, 300, 100.0, 480.0,
     8, 4, 128, 16, "7680x4320@60fps"},
    {12, "6.2", 35651584, 9951, 9951, 4278190080, 4379443200, 300, 160.0, 800.0,
     8, 4, 128, 16, "7680x4320@120fps"},
    {13, "6.3", 35651584, 9951, 9951, 4278190080, 4706009088, 300, 160.0, 800.0,
     8, 4, 128, 16, "7680x4320@120fps"},
    {14, "7.0", 142606336, 19902, 19902, 4278190080, 4706009088, 960, 160.0,
     800.0, 8, 4, 256, 32, "15360x8640@30fps"},
    {15, "7.1", 142606336, 19902, 19902, 8556380160, 8758886400, 960, 200.0,
     960.0, 8, 4, 256, 32, "15360x8640@60fps"},
    {16, "7.2", 142606336, 19902, 19902, 17112760320, 17517772800, 960, 320.0,
     1600.0, 8, 4, 256, 32, "15360x8640@120fps"},
    {17, "7.3", 142606336, 19902, 19902, 17112760320, 18824036352, 960, 320.0,
     1600.0, 8, 4, 256, 32, "15360x8640@120fps"},
    {18, "8.0", 530841600, 38400, 38400, 17112760320, 18824036352, 960, 320.0,
     1600.0, 8, 4, 512, 64, "30720x17280@30fps"},
    {19, "8.1", 530841600, 38400, 38400, 34225520640, 34910031052, 960, 400.0,
     1920.0, 8, 4, 512, 64, "30720x17280@60fps"},
    {20, "8.2", 530841600, 38400, 38400, 68451041280, 69820062105, 960, 640.0,
     3200.0, 8, 4, 512, 64, "30720x17280@120fps"},
    {21, "8.3", 530841600, 38400, 38400, 68451041280, 75296145408, 960, 640.0,
     3200.0, 8, 4, 512, 64, "30720x17280@120fps"},
}};

// §9.2, p622. Indexed [seq_tier][seq_level_idx]. `reserved` is represented by
// {0, 0, true}; note the existing planner uses scaling factors for level 31
// only as a special case (no limits), which the validator handles separately.
struct TileScaleEntry {
  int width;
  int area;
  bool reserved;
};

constexpr std::array<std::array<TileScaleEntry, 31>, 2> kTileScale = {{
    // tier 0
    {{
        {4, 4, false},   {4, 4, false},   {4, 4, false},   {4, 4, false},
        {4, 4, false},   {4, 4, false},   {4, 4, false},   {4, 4, false},
        {4, 4, false},   {4, 4, false},   {4, 4, false},   {4, 4, false},
        {4, 4, false},   {4, 4, false},   {8, 8, false},   {8, 8, false},
        {8, 8, false},   {8, 8, false},   {8, 16, false},  {8, 16, false},
        {8, 16, false},  {8, 16, false},  {0, 0, true},    {0, 0, true},
        {0, 0, true},    {0, 0, true},    {0, 0, true},    {0, 0, true},
        {0, 0, true},    {0, 0, true},    {0, 0, true},
    }},
    // tier 1
    {{
        {4, 4, false},   {4, 4, false},   {4, 4, false},   {4, 4, false},
        {4, 4, false},   {4, 4, false},   {4, 4, false},   {4, 4, false},
        {4, 4, false},   {4, 4, false},   {4, 4, false},   {4, 4, false},
        {4, 4, false},   {4, 4, false},   {8, 16, false},  {8, 16, false},
        {8, 16, false},  {8, 16, false},  {16, 32, false}, {16, 32, false},
        {16, 32, false}, {16, 32, false}, {0, 0, true},    {0, 0, true},
        {0, 0, true},    {0, 0, true},    {0, 0, true},    {0, 0, true},
        {0, 0, true},    {0, 0, true},    {0, 0, true},
    }},
}};

// ---------------------------------------------------------------------------
// Tables A.5/A.6 (p1104)
// ---------------------------------------------------------------------------

constexpr std::array<MultiSeqConfigInfo, 4> kMultiSeq = {{
    {0, "C_Main_420_10", {0, 1, 2, 31, 0}, 4, {kChroma400, kChroma420, 0, 0},
     2, false},
    {1, "C_Main_422_10", {0, 1, 2, 3, 31}, 5,
     {kChroma400, kChroma420, kChroma422, 0}, 3, false},
    {2, "C_Main_444_10", {0, 1, 2, 4, 31}, 5,
     {kChroma400, kChroma420, kChroma444, 0}, 3, false},
    // 3..63 Reserved (Table A.5).
    {3, "Reserved", {0, 0, 0, 0, 0}, 0, {0, 0, 0, 0}, 0, true},
}};

// ---------------------------------------------------------------------------
// Tables A.3/A.4 (p1102-1103)
// ---------------------------------------------------------------------------

constexpr std::array<InteropPointInfo, 5> kInterop = {{
    // iop, ext_min, ext_max, emb_min, emb_max, combination, max_layers
    {0, 1, 4, 1, 1, false, 4, false, false},
    {1, 1, 4, 1, 2, false, 4, false, false},
    {2, 1, 4, 1, 3, true, 8, false, false},
    // 3..14 Reserved (one representative row; interop_is_reserved covers all)
    {3, 0, 0, 0, 0, false, 0, true, false},
    {15, 1, 31, 1, 8, true, 248, false, true},
}};

// Table A.4 rows, keyed by (iop, ext>1, emb>1). IOP 0 has "N/A" in the
// embedded column: the rule is unchanged whether more_than_one_embedded is
// true or false, which the lookup below encodes by ignoring it.
struct InteropObuRow {
  int iop;
  bool ext_gt1;
  bool emb_gt1;
  ObuRequirement msdo;
  ObuRequirement lcr;
  bool lcr_global_or_local;
};

constexpr std::array<InteropObuRow, 8> kInteropObuRows = {{
    {0, false, false, ObuRequirement::kProhibited, ObuRequirement::kOptional,
     false},
    {0, true, false, ObuRequirement::kRequired, ObuRequirement::kOptional,
     false},
    {1, false, false, ObuRequirement::kProhibited, ObuRequirement::kOptional,
     false},
    {1, true, false, ObuRequirement::kRequired, ObuRequirement::kOptional,
     false},
    {1, false, true, ObuRequirement::kProhibited, ObuRequirement::kRequired,
     false},  // Required (Local)
    {2, false, false, ObuRequirement::kProhibited, ObuRequirement::kOptional,
     false},
    {2, true, false, ObuRequirement::kOptional, ObuRequirement::kOptional,
     true},  // One of MSDO or Global LCR
    {2, false, true, ObuRequirement::kProhibited, ObuRequirement::kRequired,
     true},  // Required (Global or Local)
}};

}  // namespace

// ---------------------------------------------------------------------------
// Profiles
// ---------------------------------------------------------------------------

int profile_table_size() { return static_cast<int>(kProfiles.size()); }

const ProfileInfo& profile_at(int index) {
  const std::size_t i =
      static_cast<std::size_t>(index < 0 ? 0 : (index >= 32 ? 31 : index));
  return kProfiles[i];
}

const ProfileInfo* find_profile(int seq_profile_idc) {
  if (seq_profile_idc < 0 || seq_profile_idc >= 32) {
    return nullptr;
  }
  return &kProfiles[static_cast<std::size_t>(seq_profile_idc)];
}

bool profile_allows_chroma(const ProfileInfo& profile, int chroma_format_idc) {
  for (int i = 0; i < profile.num_chroma_formats; ++i) {
    if (profile.chroma_format_idcs[i] == chroma_format_idc) {
      return true;
    }
  }
  return false;
}

bool profile_scaling(int seq_profile_idc, int* profile_scaling_factor,
                     int* pic_size_profile_factor,
                     double* bitrate_profile_factor) {
  const ProfileInfo* profile = find_profile(seq_profile_idc);
  if (profile == nullptr || profile->configurable) {
    return false;
  }
  if (profile_scaling_factor != nullptr) {
    *profile_scaling_factor = profile->profile_scaling_factor;
  }
  if (pic_size_profile_factor != nullptr) {
    *pic_size_profile_factor = profile->pic_size_profile_factor;
  }
  if (bitrate_profile_factor != nullptr) {
    *bitrate_profile_factor = profile->bitrate_profile_factor;
  }
  return true;
}

const char* profile_name(int seq_profile_idc) {
  const ProfileInfo* profile = find_profile(seq_profile_idc);
  if (profile == nullptr) {
    return "profile (out of range)";
  }
  return profile->profile_label;
}

// ---------------------------------------------------------------------------
// Levels
// ---------------------------------------------------------------------------

int level_table_size() { return static_cast<int>(kLevels.size()); }

const LevelInfo& level_at(int index) {
  const std::size_t i = static_cast<std::size_t>(
      index < 0 ? 0 : (index >= static_cast<int>(kLevels.size())
                           ? static_cast<int>(kLevels.size()) - 1
                           : index));
  return kLevels[i];
}

const LevelInfo* find_level(int seq_level_idx) {
  if (seq_level_idx < 0 || seq_level_idx >= static_cast<int>(kLevels.size())) {
    return nullptr;
  }
  return &kLevels[static_cast<std::size_t>(seq_level_idx)];
}

bool level_is_reserved(int seq_level_idx) {
  return seq_level_idx >= 22 && seq_level_idx <= 30;
}

bool level_is_maximum(int seq_level_idx) { return seq_level_idx == 31; }

const char* level_name(int seq_level_idx) {
  if (level_is_reserved(seq_level_idx)) {
    return "Reserved";
  }
  if (level_is_maximum(seq_level_idx)) {
    return "Max";
  }
  return level_at(seq_level_idx).name;
}

TileScaleFactor tile_scale_factor(int seq_level_idx, int seq_tier) {
  TileScaleFactor factor;
  if (seq_tier < 0 || seq_tier > 1 || seq_level_idx < 0 ||
      seq_level_idx > 30) {
    factor.reserved = true;
    return factor;
  }
  // seq_level_idx == 31 is handled by the caller: §5.18.7.3 uses sbCols /
  // total SB area instead of a scaling factor.
  const TileScaleEntry& entry =
      kTileScale[static_cast<std::size_t>(seq_tier)]
                [static_cast<std::size_t>(seq_level_idx)];
  factor.width = entry.width;
  factor.area = entry.area;
  factor.reserved = entry.reserved;
  return factor;
}

// ---------------------------------------------------------------------------
// Multi-sequence configurations
// ---------------------------------------------------------------------------

int multi_seq_table_size() { return static_cast<int>(kMultiSeq.size()); }

const MultiSeqConfigInfo& multi_seq_at(int index) {
  const std::size_t i = static_cast<std::size_t>(
      index < 0 ? 0 : (index >= static_cast<int>(kMultiSeq.size())
                           ? static_cast<int>(kMultiSeq.size()) - 1
                           : index));
  return kMultiSeq[i];
}

const MultiSeqConfigInfo* find_multi_seq_config(int lcr_config_idc) {
  if (lcr_config_idc < 0 || lcr_config_idc > 2) {
    return nullptr;
  }
  return &kMultiSeq[static_cast<std::size_t>(lcr_config_idc)];
}

bool multi_seq_is_reserved(int lcr_config_idc) {
  return lcr_config_idc >= 3 && lcr_config_idc <= 63;
}

bool multi_seq_allows_profile(const MultiSeqConfigInfo& config, int profile) {
  for (int i = 0; i < config.num_profiles; ++i) {
    if (config.profiles[i] == profile) {
      return true;
    }
  }
  return false;
}

bool multi_seq_allows_chroma(const MultiSeqConfigInfo& config, int chroma) {
  for (int i = 0; i < config.num_chroma_formats; ++i) {
    if (config.chroma_format_idcs[i] == chroma) {
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Interoperability points
// ---------------------------------------------------------------------------

int interop_table_size() { return static_cast<int>(kInterop.size()); }

const InteropPointInfo& interop_at(int index) {
  const std::size_t i = static_cast<std::size_t>(
      index < 0 ? 0 : (index >= static_cast<int>(kInterop.size())
                           ? static_cast<int>(kInterop.size()) - 1
                           : index));
  return kInterop[i];
}

const InteropPointInfo* find_interop_point(int iop) {
  if (iop < 0 || iop > 15) {
    return nullptr;
  }
  if (iop <= 2) {
    return &kInterop[static_cast<std::size_t>(iop)];
  }
  if (iop == 15) {
    return &kInterop[4];
  }
  return &kInterop[3];  // reserved row
}

bool interop_is_reserved(int iop) { return iop >= 3 && iop <= 14; }

const char* interop_name(int iop) {
  if (iop == 15) {
    return "IOP 15 (max)";
  }
  if (interop_is_reserved(iop)) {
    return "IOP (reserved)";
  }
  return iop >= 0 && iop <= 2 ? "IOP 0/1/2" : "IOP (out of range)";
}

InteropObuRule interop_obu_rule(int iop, bool more_than_one_extended,
                                bool more_than_one_embedded) {
  InteropObuRule rule;
  if (iop < 0 || iop > 2) {
    return rule;
  }
  // Table A.4 does not distinguish the "ext/emb > 1" combinations for IOP 2
  // when both are true: "MSDO plus Local LCR, or Global LCR". The row marked
  // with lcr_global_or_local covers it.
  const bool emb = iop == 0 ? false : more_than_one_embedded;
  if (iop == 2 && more_than_one_extended && emb) {
    rule.msdo = ObuRequirement::kOptional;
    rule.lcr = ObuRequirement::kRequired;
    rule.lcr_global_or_local = true;
    return rule;
  }
  for (const InteropObuRow& row : kInteropObuRows) {
    if (row.iop == iop && row.ext_gt1 == more_than_one_extended &&
        row.emb_gt1 == emb) {
      rule.msdo = row.msdo;
      rule.lcr = row.lcr;
      rule.lcr_global_or_local = row.lcr_global_or_local;
      return rule;
    }
  }
  return rule;
}

}  // namespace av2enc::conformance
