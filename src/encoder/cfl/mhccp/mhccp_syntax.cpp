// read_cfl_alphas() writer/reader for MHCCP (§5.20.7.32, §6.19.7.25).

#include "mhccp_syntax.h"

#include "entropy/av2/av2_entropy.h"

#include <cstring>

namespace av2enc::cfl::mhccp {
namespace {

void copy_array(std::uint16_t* dst, const std::uint16_t* src, int n) {
  std::memcpy(dst, src, static_cast<std::size_t>(n) * sizeof(std::uint16_t));
}

bool fail(std::string* error, const char* what) {
  if (error != nullptr) {
    *error = what;
  }
  return false;
}

bool sym_encode(av2::Encoder& encoder, std::uint16_t* cdf, int n, int symbol,
                std::string* error) {
  const av2::Status status = encoder.encode_symbol(cdf, n, symbol);
  if (status != av2::Status::kOk) {
    if (error != nullptr) {
      *error = av2::status_name(status);
    }
    return false;
  }
  return true;
}

bool sym_decode(av2::Decoder& decoder, std::uint16_t* cdf, int n, int* symbol,
                std::string* error) {
  const av2::Status status = decoder.decode_symbol(cdf, n, symbol);
  if (status != av2::Status::kOk) {
    if (error != nullptr) {
      *error = av2::status_name(status);
    }
    return false;
  }
  return true;
}

int sign_u_from_joint(int signs) { return (signs + 1) / 3; }
int sign_v_from_joint(int signs) { return (signs + 1) % 3; }

}  // namespace

void CflCdfs::init() {
  copy_array(&is_cfl[0][0], &kIsCflCdf[0][0], 3 * 3);
  copy_array(cfl_index, kCflIndexCdf, 3);
  copy_array(cfl_sign, kCflSignCdf, 9);
  copy_array(&cfl_alpha[0][0], &kCflAlphaCdf[0][0], 6 * 9);
  copy_array(cfl_mhccp, kCflMhccpCdf, 3);
  copy_array(&cfl_mh_dir[0][0], &kCflMhDirCdf[0][0], 4 * 4);
}

CflSyntax make_explicit_syntax(const AlphaPair& alpha) {
  const Signalling signalling = explicit_signalling(alpha);
  CflSyntax syntax{};
  syntax.cfl_mhccp = 0;
  syntax.cfl_index = kIndexExplicit;
  syntax.cfl_alpha_signs = signalling.alpha_signs;
  syntax.cfl_alpha_u = signalling.alpha_u;
  syntax.cfl_alpha_v = signalling.alpha_v;
  return syntax;
}

CflSyntax make_derived_syntax() {
  CflSyntax syntax{};
  syntax.cfl_mhccp = 0;
  syntax.cfl_index = kIndexDerivedAlpha;
  return syntax;
}

CflSyntax make_multi_syntax(int cfl_mh_dir) {
  CflSyntax syntax{};
  syntax.cfl_mhccp = 1;
  syntax.cfl_index = kIndexMulti;
  syntax.cfl_mh_dir = cfl_mh_dir;
  return syntax;
}

bool valid_syntax(const CflSyntax& syntax, bool enable_cfl_intra,
                  bool mhccp_allowed) {
  if (syntax.cfl_index < kIndexExplicit || syntax.cfl_index > kIndexMulti) {
    return false;
  }
  if (!enable_cfl_intra) {
    if (syntax.cfl_index != kIndexMulti || syntax.cfl_mhccp != 1) {
      return false;
    }
  } else if (mhccp_allowed) {
    if (syntax.cfl_mhccp != (syntax.cfl_index == kIndexMulti ? 1 : 0)) {
      return false;
    }
  } else if (syntax.cfl_mhccp != 0 || syntax.cfl_index == kIndexMulti) {
    return false;
  }

  if (syntax.cfl_index == kIndexMulti) {
    return syntax.cfl_mh_dir >= 0 && syntax.cfl_mh_dir < kDirections;
  }
  if (syntax.cfl_index != kIndexExplicit) {
    return true;
  }

  if (syntax.cfl_alpha_signs < 0 || syntax.cfl_alpha_signs >= kJointSigns) {
    return false;
  }
  const int sign_u = sign_u_from_joint(syntax.cfl_alpha_signs);
  const int sign_v = sign_v_from_joint(syntax.cfl_alpha_signs);
  if ((sign_u == kSignZero && sign_v == kSignZero) ||
      syntax.cfl_alpha_u < 0 || syntax.cfl_alpha_u >= kAlphabetSize ||
      syntax.cfl_alpha_v < 0 || syntax.cfl_alpha_v >= kAlphabetSize) {
    return false;
  }
  // Fields not coded by read_cfl_alphas() are canonicalised to zero.
  if (sign_u == kSignZero && syntax.cfl_alpha_u != 0) {
    return false;
  }
  if (sign_v == kSignZero && syntax.cfl_alpha_v != 0) {
    return false;
  }
  return true;
}

bool write_cfl_alphas(av2::Encoder& encoder, CflCdfs& cdfs,
                      const CflSyntax& syntax, bool enable_cfl_intra,
                      bool mhccp_allowed, int size_group,
                      std::string* error) {
  if (error != nullptr) {
    error->clear();
  }
  if (size_group < 0 || size_group >= 4) {
    return fail(error, "cfl_mh_dir size_group out of range");
  }
  if (!valid_syntax(syntax, enable_cfl_intra, mhccp_allowed)) {
    return fail(error, "inconsistent CfL/MHCCP syntax");
  }

  if (enable_cfl_intra) {
    if (mhccp_allowed) {
      if (!sym_encode(encoder, cdfs.cfl_mhccp, 2, syntax.cfl_mhccp, error)) {
        return false;
      }
    }
    if (syntax.cfl_index != kIndexMulti) {
      if (!sym_encode(encoder, cdfs.cfl_index, 2, syntax.cfl_index, error)) {
        return false;
      }
    }
  }
  if (syntax.cfl_index == kIndexMulti) {
    if (!sym_encode(encoder, cdfs.cfl_mh_dir[size_group], 3,
                    syntax.cfl_mh_dir, error)) {
      return false;
    }
  }
  if (syntax.cfl_index != kIndexExplicit) {
    return true;
  }

  if (!sym_encode(encoder, cdfs.cfl_sign, kJointSigns,
                  syntax.cfl_alpha_signs, error)) {
    return false;
  }
  const int sign_u = sign_u_from_joint(syntax.cfl_alpha_signs);
  const int sign_v = sign_v_from_joint(syntax.cfl_alpha_signs);
  if (sign_u != kSignZero) {
    const int ctx_u = (sign_u - 1) * 3 + sign_v;
    if (!sym_encode(encoder, cdfs.cfl_alpha[ctx_u], kAlphabetSize,
                    syntax.cfl_alpha_u, error)) {
      return false;
    }
  }
  if (sign_v != kSignZero) {
    const int ctx_v = (sign_v - 1) * 3 + sign_u;
    if (!sym_encode(encoder, cdfs.cfl_alpha[ctx_v], kAlphabetSize,
                    syntax.cfl_alpha_v, error)) {
      return false;
    }
  }
  return true;
}

bool read_cfl_alphas(av2::Decoder& decoder, CflCdfs& cdfs,
                     bool enable_cfl_intra, bool mhccp_allowed,
                     int size_group, CflSyntax* syntax, std::string* error) {
  if (syntax == nullptr) {
    return fail(error, "null syntax output");
  }
  if (error != nullptr) {
    error->clear();
  }
  if (size_group < 0 || size_group >= 4) {
    return fail(error, "cfl_mh_dir size_group out of range");
  }

  CflSyntax out{};
  if (!enable_cfl_intra) {
    out.cfl_mhccp = 1;
  } else if (mhccp_allowed) {
    int bit = 0;
    if (!sym_decode(decoder, cdfs.cfl_mhccp, 2, &bit, error)) {
      return false;
    }
    out.cfl_mhccp = bit;
  } else {
    out.cfl_mhccp = 0;
  }

  if (out.cfl_mhccp != 0) {
    out.cfl_index = kIndexMulti;
  } else {
    int index = 0;
    if (!sym_decode(decoder, cdfs.cfl_index, 2, &index, error)) {
      return false;
    }
    out.cfl_index = index;
  }

  if (out.cfl_index == kIndexMulti) {
    int dir = 0;
    if (!sym_decode(decoder, cdfs.cfl_mh_dir[size_group], 3, &dir, error)) {
      return false;
    }
    out.cfl_mh_dir = dir;
  }

  if (out.cfl_index == kIndexExplicit) {
    int signs = 0;
    if (!sym_decode(decoder, cdfs.cfl_sign, kJointSigns, &signs, error)) {
      return false;
    }
    out.cfl_alpha_signs = signs;
    const int sign_u = sign_u_from_joint(signs);
    const int sign_v = sign_v_from_joint(signs);
    if (sign_u != kSignZero) {
      const int ctx_u = (sign_u - 1) * 3 + sign_v;
      int value = 0;
      if (!sym_decode(decoder, cdfs.cfl_alpha[ctx_u], kAlphabetSize, &value,
                      error)) {
        return false;
      }
      out.cfl_alpha_u = value;
    }
    if (sign_v != kSignZero) {
      const int ctx_v = (sign_v - 1) * 3 + sign_u;
      int value = 0;
      if (!sym_decode(decoder, cdfs.cfl_alpha[ctx_v], kAlphabetSize, &value,
                      error)) {
        return false;
      }
      out.cfl_alpha_v = value;
    }
  }

  *syntax = out;
  return true;
}

}  // namespace av2enc::cfl::mhccp
