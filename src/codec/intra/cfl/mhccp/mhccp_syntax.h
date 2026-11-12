#ifndef AV2ENC_ENCODER_CFL_MHCCP_SYNTAX_H_
#define AV2ENC_ENCODER_CFL_MHCCP_SYNTAX_H_

// read_cfl_alphas() syntax for cfl_index == CFL_MULTI.
//
// AV2 v1.0.0 §5.20.7.32 (syntax) and §6.19.7.25 (semantics). This module is
// self-contained: it owns the six default CfL CDFs needed by the syntax,
// including Default_Cfl_Mhccp_Cdf and Default_Cfl_Mh_Dir_Cdf, which have no
// generated C++ header in this tree. It writes/reads §8.2 symbols but does not
// assemble a frame; see README.md for the exact core call sites that would be
// required to put these symbols into a decodable stream.

#include "mhccp_cdf_tables.h"
#include "mhccp_common.h"

#include "../cfl_common.h"

#include <cstdint>
#include <string>

namespace av2enc::av2 {
struct Encoder;
struct Decoder;
}  // namespace av2enc::av2

namespace av2enc::cfl::mhccp {

// One read_cfl_alphas() result. Fields not coded by the selected cfl_index are
// left at zero and ignored.
struct CflSyntax {
  int cfl_mhccp = 0;  // S() flag when enable_cfl_intra && is_mhccp_allowed()
  int cfl_index = kIndexExplicit;
  int cfl_mh_dir = 0;  // CFL_MULTI only, 0..2
  int cfl_alpha_signs = 0;
  int cfl_alpha_u = 0;
  int cfl_alpha_v = 0;
};

// Per-tile CDF state, initialised from §9.3 defaults.
struct CflCdfs {
  std::uint16_t is_cfl[3][3]{};
  std::uint16_t cfl_index[3]{};
  std::uint16_t cfl_sign[9]{};
  std::uint16_t cfl_alpha[6][9]{};
  std::uint16_t cfl_mhccp[3]{};
  std::uint16_t cfl_mh_dir[4][4]{};

  void init();
};

// Constructors for the three hypotheses.
CflSyntax make_explicit_syntax(const AlphaPair& alpha);
CflSyntax make_derived_syntax();
CflSyntax make_multi_syntax(int cfl_mh_dir);

// True when the fields are consistent with read_cfl_alphas() for the given
// sequence/tool state.
bool valid_syntax(const CflSyntax& syntax, bool enable_cfl_intra,
                  bool mhccp_allowed);

// Parse/emit read_cfl_alphas(). `size_group` is Size_Group[MiSize] (0..3) and
// selects TileCflMhDirCdf; use size_group_for_chroma_block() for 4:2:0. The
// coder's disable_cdf_update setting is part of the av2::Encoder/Decoder.
bool write_cfl_alphas(av2::Encoder& encoder, CflCdfs& cdfs,
                      const CflSyntax& syntax, bool enable_cfl_intra,
                      bool mhccp_allowed, int size_group,
                      std::string* error);
bool read_cfl_alphas(av2::Decoder& decoder, CflCdfs& cdfs,
                     bool enable_cfl_intra, bool mhccp_allowed,
                     int size_group, CflSyntax* syntax, std::string* error);

}  // namespace av2enc::cfl::mhccp

#endif  // AV2ENC_ENCODER_CFL_MHCCP_SYNTAX_H_
