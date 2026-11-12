#ifndef AV2ENC_ENCODER_CFL_DS_FILTERS_GOLDENS_H_
#define AV2ENC_ENCODER_CFL_DS_FILTERS_GOLDENS_H_

// Pinned §7.13.5 prediction goldens for every down-sampling filter.
//
// Regenerate with:
//   build/ds_filters_test --print-goldens > /tmp/goldens.txt
// and replace the two arrays below.  A change here means the filter
// arithmetic, the edge handling or the test fixture changed; review the diff
// before accepting it.

#include <cstdint>

namespace av2enc::cfl::ds::test {

struct Golden {
  const char* group;
  const char* name;
  int filter;
  std::uint64_t l_crc;            // FNV-1a of array L, 3 fractional bits
  int derived_alpha;              // §7.13.5 implicitAlpha (Q8)
  std::uint64_t pred_fixed_crc;   // prediction with alpha = +2 eighths
  std::uint64_t pred_derived_crc; // prediction with derived alpha
};

inline constexpr Golden kSyntheticGoldens[] {
  {"synthetic", "synth_uniform_8x8", 0, 0xbe64e94d0c76a783ULL, 0, 0x4e92b3e665278843ULL, 0x4e92b3e665278843ULL},
  {"synthetic", "synth_uniform_8x8", 1, 0xbe64e94d0c76a783ULL, 0, 0x4e92b3e665278843ULL, 0x4e92b3e665278843ULL},
  {"synthetic", "synth_uniform_8x8", 2, 0xbe64e94d0c76a783ULL, 0, 0x4e92b3e665278843ULL, 0x4e92b3e665278843ULL},
  {"synthetic", "synth_uniform_8x8", 3, 0xbe64e94d0c76a783ULL, 0, 0x4e92b3e665278843ULL, 0x4e92b3e665278843ULL},
  {"synthetic", "synth_no_neighbours_4x4", 0, 0x9fa8da8ba04d7198ULL, 0, 0x6fffddc300e268d7ULL, 0x02301c1fff61b603ULL},
  {"synthetic", "synth_no_neighbours_4x4", 1, 0xd23b4c615565a17fULL, 0, 0x4b68d900bce30de0ULL, 0x02301c1fff61b603ULL},
  {"synthetic", "synth_no_neighbours_4x4", 2, 0xe66e371bb990d9f0ULL, 0, 0x38530356b8f18203ULL, 0x02301c1fff61b603ULL},
  {"synthetic", "synth_no_neighbours_4x4", 3, 0x9fa8da8ba04d7198ULL, 0, 0x6fffddc300e268d7ULL, 0x02301c1fff61b603ULL},
  {"synthetic", "synth_gradient_left_edge_4x8", 0, 0xa7963e1f682bc82dULL, 51, 0x1e17b58bb5615973ULL, 0x57aefaf7b94bb307ULL},
  {"synthetic", "synth_gradient_left_edge_4x8", 1, 0x81db963ef6391bbaULL, 51, 0xefe479c3e044158bULL, 0xb67815f05a151470ULL},
  {"synthetic", "synth_gradient_left_edge_4x8", 2, 0x58927d7661313958ULL, 50, 0x53c0266c3547d4f4ULL, 0x75e5ca09aab5d2dcULL},
  {"synthetic", "synth_gradient_left_edge_4x8", 3, 0xa7963e1f682bc82dULL, 51, 0x1e17b58bb5615973ULL, 0x57aefaf7b94bb307ULL},
  {"synthetic", "synth_sb_top_16x8", 0, 0xff2b4bde22d85189ULL, 2, 0x9408f78c882e290fULL, 0x76b7f98225ac34abULL},
  {"synthetic", "synth_sb_top_16x8", 1, 0x0423bcaa0db3fbc5ULL, 1, 0x4d3dc7a24e7fdbd5ULL, 0xf5062c6f8e68dc68ULL},
  {"synthetic", "synth_sb_top_16x8", 2, 0xd92818ed39346018ULL, 1, 0x7717d9d3b22cf2c1ULL, 0xf5062c6f8e68dc68ULL},
  {"synthetic", "synth_sb_top_16x8", 3, 0xff2b4bde22d85189ULL, 2, 0x9408f78c882e290fULL, 0x76b7f98225ac34abULL},
  {"synthetic", "synth_partial_right_bottom_8x8", 0, 0x6274c18ce33e4a6aULL, -64, 0x269f81a2044a4f03ULL, 0xc56791842701c037ULL},
  {"synthetic", "synth_partial_right_bottom_8x8", 1, 0x5518d55a175d38f7ULL, -64, 0x760800be7f192b63ULL, 0x6f6d0a6539447087ULL},
  {"synthetic", "synth_partial_right_bottom_8x8", 2, 0x98de89d0ae226518ULL, -64, 0x948668a2d070068aULL, 0x650ac93ba9cb21dcULL},
  {"synthetic", "synth_partial_right_bottom_8x8", 3, 0x6274c18ce33e4a6aULL, -64, 0x269f81a2044a4f03ULL, 0xc56791842701c037ULL},
};

inline constexpr Golden kRealGoldens[] {
  {"real", "real_64x64_at_100_80", 0, 0x2a192cd1eb600383ULL, 0, 0xd535e0036eca8b45ULL, 0xd535e0036eca8b45ULL},
  {"real", "real_64x64_at_100_80", 1, 0x2a192cd1eb600383ULL, 0, 0xd535e0036eca8b45ULL, 0xd535e0036eca8b45ULL},
  {"real", "real_64x64_at_100_80", 2, 0x2a192cd1eb600383ULL, 0, 0xd535e0036eca8b45ULL, 0xd535e0036eca8b45ULL},
  {"real", "real_64x64_at_100_80", 3, 0x2a192cd1eb600383ULL, 0, 0xd535e0036eca8b45ULL, 0xd535e0036eca8b45ULL},
  {"real", "real_32x16_at_300_150", 0, 0xf247da2871dfb462ULL, -139, 0x7c2ba0c8629d48d0ULL, 0x5c7bc6b6242ad997ULL},
  {"real", "real_32x16_at_300_150", 1, 0x88ca0fdf50778668ULL, -132, 0xaa427eed82323b69ULL, 0xbdd4dd585329e9d0ULL},
  {"real", "real_32x16_at_300_150", 2, 0x91741970095a096fULL, -114, 0x401c1dbf2c2296c7ULL, 0x002ad279896c1f41ULL},
  {"real", "real_32x16_at_300_150", 3, 0xf247da2871dfb462ULL, -139, 0x7c2ba0c8629d48d0ULL, 0x5c7bc6b6242ad997ULL},
  {"real", "real_16x8_at_320_180", 0, 0x2e0c9043c6f93859ULL, -143, 0x7714a061e40e159fULL, 0x64f20e8dbf6f3474ULL},
  {"real", "real_16x8_at_320_180", 1, 0xdc5a824ca99c8224ULL, -149, 0x9cd64da10c3213baULL, 0x3b7d0449d1c9690dULL},
  {"real", "real_16x8_at_320_180", 2, 0x76efec0373130d09ULL, -149, 0x46497b91d1d5fd55ULL, 0x0ccb09cd5d2c6613ULL},
  {"real", "real_16x8_at_320_180", 3, 0x2e0c9043c6f93859ULL, -143, 0x7714a061e40e159fULL, 0x64f20e8dbf6f3474ULL},
  {"real", "real_8x8_at_400_200", 0, 0x9506d7e0c0374266ULL, -133, 0xe541a42941b8a1dcULL, 0x5b890a028a0ba89cULL},
  {"real", "real_8x8_at_400_200", 1, 0x26cf69547c4daefcULL, -141, 0x0a170a48f9787a46ULL, 0x6bb7b1bfae8b257cULL},
  {"real", "real_8x8_at_400_200", 2, 0xa02ccf6acad8f408ULL, -136, 0x31320c138ba4fbb4ULL, 0x4d926ec230155030ULL},
  {"real", "real_8x8_at_400_200", 3, 0x9506d7e0c0374266ULL, -133, 0xe541a42941b8a1dcULL, 0x5b890a028a0ba89cULL},
  {"real", "real_4x4_at_500_250", 0, 0x5fec38cf6a564683ULL, 0, 0xd535e0036eca8b45ULL, 0xd535e0036eca8b45ULL},
  {"real", "real_4x4_at_500_250", 1, 0x5fec38cf6a564683ULL, 0, 0xd535e0036eca8b45ULL, 0xd535e0036eca8b45ULL},
  {"real", "real_4x4_at_500_250", 2, 0x5fec38cf6a564683ULL, 0, 0xd535e0036eca8b45ULL, 0xd535e0036eca8b45ULL},
  {"real", "real_4x4_at_500_250", 3, 0x5fec38cf6a564683ULL, 0, 0xd535e0036eca8b45ULL, 0xd535e0036eca8b45ULL},
};

}  // namespace av2enc::cfl::ds::test

#endif  // AV2ENC_ENCODER_CFL_DS_FILTERS_GOLDENS_H_
