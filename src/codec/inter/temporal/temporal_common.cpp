// AV2 v1.0.0 temporal inter tools — small shared helpers.

#include "codec/inter/temporal/temporal_types.h"

namespace av2enc::inter::temporal {

const char* temporal_status_string(TemporalStatus status) {
  switch (status) {
    case TemporalStatus::kOk:
      return "ok";
    case TemporalStatus::kInvalidArgument:
      return "invalid argument";
    case TemporalStatus::kUnsupported:
      return "unsupported";
  }
  return "unknown";
}

}  // namespace av2enc::inter::temporal
