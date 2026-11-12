// Normative default CDFs used by read_cfl_alphas() (§9.3).
//
// Source: spec/tables/default_cdf.json (AV2 v1.0.0, commit 38f28e7).
// There is no generated C++ header for Default_Cfl_Mhccp_Cdf or
// Default_Cfl_Mh_Dir_Cdf in this tree, so the two arrays are copied here
// together with the CfL arrays already used by M1.
#pragma once

namespace av2enc::cfl::mhccp {

// Default_Is_Cfl_Cdf [CFL_CONTEXTS=3][3]
inline constexpr unsigned short kIsCflCdf[3][3] = {
    {20441, 30, 0}, {11610, 6, 0}, {4643, 0, 0},
};

// Default_Cfl_Index_Cdf [3] — symbols CFL_EXPLICIT / CFL_DERIVED_ALPHA.
inline constexpr unsigned short kCflIndexCdf[3] = {
    12507, 30, 0,
};

// Default_Cfl_Sign_Cdf [CFL_JOINT_SIGNS + 1 = 9]
inline constexpr unsigned short kCflSignCdf[9] = {
    2421, 4332, 11256, 12766, 21386, 28725, 32087, 62, 0,
};

// Default_Cfl_Alpha_Cdf [CFL_ALPHA_CONTEXTS=6][CFL_ALPHABET_SIZE + 1 = 9]
inline constexpr unsigned short kCflAlphaCdf[6][9] = {
    {21679, 25305, 30646, 31512, 32537, 32646, 32696, 62, 0},
    {8262, 16302, 24082, 29422, 31398, 32286, 32525, 62, 0},
    {17235, 26166, 30378, 31305, 32373, 32549, 32668, 62, 0},
    {17618, 25732, 27865, 30338, 31125, 31522, 32238, 62, 0},
    {17542, 23066, 27907, 28728, 30702, 31165, 31435, 62, 0},
    {17675, 24802, 30468, 30783, 31841, 32264, 32422, 62, 0},
};

// Default_Cfl_Mhccp_Cdf [3]
inline constexpr unsigned short kCflMhccpCdf[3] = {
    15499, 30, 0,
};

// Default_Cfl_Mh_Dir_Cdf [BLOCK_SIZE_GROUPS=4][3 + 1 = 4]
inline constexpr unsigned short kCflMhDirCdf[4][4] = {
    {10923, 21845, 0, 0},
    {8795, 15105, 6, 0},
    {10433, 15974, 32, 0},
    {17085, 21689, 32, 0},
};

// Size_Group[29] from spec/tables/block_tables.json (§9.2). Needed for the
// cfl_mh_dir context: TileCflMhDirCdf[Size_Group[MiSize]].
inline constexpr int kSizeGroup[29] = {
    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2,
};

// Block_Width/Block_Height for the 29 AV2 block sizes, in enum order
// (BLOCK_4X4 .. BLOCK_64X8). Used only to derive the Size_Group context from
// a chroma block size in 4:2:0.
inline constexpr int kBlockWidth[29] = {
    4,  4,  8,  8,  8,  16, 16, 16, 32, 32, 32, 64, 64, 64, 128,
    128, 128, 256, 256, 4, 16, 8, 32, 16, 64, 4, 32, 8, 64};
inline constexpr int kBlockHeight[29] = {
    4,  8,  4,  8,  16, 8,  16, 32, 16, 32, 64, 32, 64, 128, 64,
    128, 256, 128, 256, 16, 4, 32, 8, 64, 16, 32, 4, 64, 8};

// Size_Group context for a 4:2:0 chroma block. Returns 0..3, or -1 when the
// luma block size is not in the standard 29-entry list.
inline constexpr int size_group_for_chroma_block(int chroma_w, int chroma_h) {
  const int luma_w = chroma_w << 1;
  const int luma_h = chroma_h << 1;
  for (int i = 0; i < 29; ++i) {
    if (kBlockWidth[i] == luma_w && kBlockHeight[i] == luma_h) {
      return kSizeGroup[i];
    }
  }
  return -1;
}

}  // namespace av2enc::cfl::mhccp
