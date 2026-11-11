// AV2 v1.0.0 Annex A tables used by the encoder-side conformance validator.
//
// Spec-only: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7).
//
//   * Table A.1  profile definitions                          (§A.2, p1101-1102)
//   * Table A.2  ProfileScalingFactor / PicSizeProfileFactor /
//                BitrateProfileFactor                        (§A.2, p1102)
//   * Table A.3  interoperability points                     (§A.3, p1102)
//   * Table A.4  OBU requirements per interoperability point  (§A.3, p1103)
//   * Table A.5  multi-sequence configurations               (§A.3, p1104)
//   * Table A.6  allowed syntax element values per
//                multi-sequence configuration                (§A.3, p1104)
//   * Table A.7  seq_level_idx -> level mapping              (§A.4, p1104-1105)
//   * Table A.8  MaxPicSize / MaxHSize / MaxVSize /
//                MaxDisplayRate / MaxDecodeRate             (§A.4, p1105-1106)
//   * Table A.9  MaxHeaderRate / MainMbps / HighMbps /
//                MainCR / HighCR / MaxTiles / MaxTileCols   (§A.4, p1106)
//   * Table 9.x  Tile_Width_Scaling_Factor /
//                Tile_Area_Scaling_Factor                    (§9.2, p622)
//
// The tables are transcribed verbatim. Do not edit a value here to make a
// check pass: the only correct fix is to change the level/tier/profile the
// encoder signals (or the frame/tile plan).
//
// This component is default-free and has no dependency on the rest of the
// encoder. It emits no bits and changes no encoder behaviour.
#pragma once

#include <cstdint>

namespace av2enc::conformance {

// ---------------------------------------------------------------------------
// §A.2 Table A.1 / Table A.2 — profiles
// ---------------------------------------------------------------------------

// chroma_format_idc values are the §6.4.1 Table 6.2 numbers, not an enum:
//   0 = 4:2:0, 1 = 4:0:0, 2 = 4:4:4, 3 = 4:2:2.
struct ProfileInfo {
  int idc = 0;
  const char* label = "";
  const char* profile_label = "";
  // Interoperability point implied by the profile id (Table A.1). -1 for the
  // Configurable profile, where it is carried by the MSDO/LCR instead.
  int interop_point = -1;
  // Table A.2. Profile 31 has no defined scaling factors; the fields are
  // zeroed and `configurable` is true.
  int profile_scaling_factor = 0;
  int pic_size_profile_factor = 0;
  double bitrate_profile_factor = 0.0;
  // Allowed chroma_format_idc values (Table A.1).
  int chroma_format_idcs[4] = {0, 0, 0, 0};
  int num_chroma_formats = 0;
  bool reserved = false;
  bool configurable = false;
};

// 32 entries, index == seq_profile_idc (0..31). 5..30 are reserved.
int profile_table_size();
const ProfileInfo& profile_at(int index);
// nullptr only when seq_profile_idc is outside 0..31.
const ProfileInfo* find_profile(int seq_profile_idc);
bool profile_allows_chroma(const ProfileInfo& profile, int chroma_format_idc);
// Profile 31 has no Table A.2 row; returns false and leaves the outputs 0/-1.
bool profile_scaling(int seq_profile_idc, int* profile_scaling_factor,
                     int* pic_size_profile_factor,
                     double* bitrate_profile_factor);
// Convenience label used in messages: "profile 3 (Main_422_10_IP1)".
const char* profile_name(int seq_profile_idc);

// ---------------------------------------------------------------------------
// §A.4 Tables A.7/A.8/A.9 — levels
// ---------------------------------------------------------------------------

struct LevelInfo {
  int idx = 0;                 // seq_level_idx
  const char* name = "";       // "5.1"
  int64_t max_pic_size = 0;    // samples
  int max_hsize = 0;           // samples
  int max_vsize = 0;           // samples
  int64_t max_display_rate = 0;  // samples/sec
  int64_t max_decode_rate = 0;   // samples/sec
  int max_header_rate = 0;       // MaxHeaderRate, /sec
  // HighMbps/HighCR are undefined below level 4.0 (NOTE, p1106); the fields
  // are 0 there and seq_tier = 1 is illegal.
  double main_mbps = 0.0;
  double high_mbps = 0.0;
  int main_cr = 0;
  int high_cr = 0;
  int max_tiles = 0;
  int max_tile_cols = 0;
  const char* example = "";
  bool reserved = false;
  bool maximum_parameters = false;
};

// 22 entries, index == seq_level_idx (0..21). Tables A.8/A.9.
int level_table_size();
const LevelInfo& level_at(int index);
// nullptr for 22..30 (reserved) and for out-of-range values.
const LevelInfo* find_level(int seq_level_idx);
// 22..30 are Reserved (Table A.7, p1105).
bool level_is_reserved(int seq_level_idx);
// 31 means "maximum parameters": no level-based constraints apply (p1111).
bool level_is_maximum(int seq_level_idx);
const char* level_name(int seq_level_idx);  // "2.0" ... "8.3", "Reserved", "Max"

// §9.2 / p622. A reserved level/tier combination has reserved == true.
struct TileScaleFactor {
  int width = 0;
  int area = 0;
  bool reserved = false;
};
TileScaleFactor tile_scale_factor(int seq_level_idx, int seq_tier);

// ---------------------------------------------------------------------------
// §A.3 Tables A.5/A.6 — multi-sequence configurations
// ---------------------------------------------------------------------------

struct MultiSeqConfigInfo {
  int config_id = 0;
  const char* label = "";
  // Table A.6: allowed seq_profile_idc values. Table A.6 lists
  // "0..2, 31" (id 0), "0..3, 31" (id 1), "0..2, 4, 31" (id 2).
  int profiles[5] = {0, 0, 0, 0, 0};
  int num_profiles = 0;
  // Table A.6 allowed chroma_format_idc values.
  int chroma_format_idcs[4] = {0, 0, 0, 0};
  int num_chroma_formats = 0;
  // Table A.5/A.6: bit_depth_idc 0..1 for every defined configuration.
  bool reserved = false;
};

// 4 entries for config ids 0..2 plus the reserved row 3..63. `find` returns
// nullptr for 3..63 and for out-of-range values; use `multi_seq_is_reserved`.
int multi_seq_table_size();
const MultiSeqConfigInfo& multi_seq_at(int index);
const MultiSeqConfigInfo* find_multi_seq_config(int lcr_config_idc);
bool multi_seq_is_reserved(int lcr_config_idc);
bool multi_seq_allows_profile(const MultiSeqConfigInfo& config, int profile);
bool multi_seq_allows_chroma(const MultiSeqConfigInfo& config, int chroma);

// ---------------------------------------------------------------------------
// §A.3 Tables A.3/A.4 — interoperability points
// ---------------------------------------------------------------------------

// Table A.3. IOP 3..14 are reserved; IOP 15 is the maximum.
struct InteropPointInfo {
  int iop = 0;
  int min_extended_layers = 0;
  int max_extended_layers = 0;
  int min_embedded_layers = 0;
  int max_embedded_layers = 0;
  bool combination_allowed = false;  // Table A.3 "0 or 1"
  int max_layers = 0;
  bool reserved = false;
  bool maximum = false;
};

// 0,1,2,3..14 (one reserved row), 15. `find` returns nullptr outside 0..15.
int interop_table_size();
const InteropPointInfo& interop_at(int index);
const InteropPointInfo* find_interop_point(int iop);
bool interop_is_reserved(int iop);
const char* interop_name(int iop);

// Table A.4. `Required` LCR means Required (Local) for IOP 1, and
// Required (Global or Local) for IOP 2. `lcr_global_or_local` records that.
enum class ObuRequirement { kProhibited, kOptional, kRequired };

struct InteropObuRule {
  ObuRequirement msdo = ObuRequirement::kOptional;
  ObuRequirement lcr = ObuRequirement::kOptional;
  bool lcr_global_or_local = false;
};

// `more_than_one_extended` / `more_than_one_embedded` are the Table A.4 "> 1"
// columns. IOP 0 has N/A in the embedded column and is treated as false.
InteropObuRule interop_obu_rule(int iop, bool more_than_one_extended,
                                bool more_than_one_embedded);

}  // namespace av2enc::conformance
