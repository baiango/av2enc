#!/usr/bin/env bash
# dav2d wrapper that forces --strict 1 for the M2 RDO experiment.
#
# The M2 gate builds a decode command line and passes it here.  The real
# decoder path is supplied by REAL_DAV2D so this wrapper can be used for both
# the gate's recon comparison and tools/quality.py.
set -euo pipefail
: "${REAL_DAV2D:?REAL_DAV2D must point at the real dav2d binary}"
exec "$REAL_DAV2D" --strict 1 "$@"
