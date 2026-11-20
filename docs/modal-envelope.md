# Modal envelope offload

Status: **encode-only, local scoring**. Moves the ten-lane loop's encodes off
the Mac; the K measurement still runs in `tools/oracle.py` locally, so the
metric implementation and the gate are unchanged. The Modal wrapper keeps the
same CLI, spec, QP bracketing and JSON report fields as `tools/envelope.py`.

## Why encode-only

The remote metrics image is **not used in this build**. Two blockers:

1. There is no pinned Linux `avmdec` in the tree (`tools/bin/avmdec` is a
   macOS binary). The local oracle's contract is the AVM decoder; a Linux
   decoder would be a new tooling pin with its own parity check.
2. The first `WITH_METRICS=1` attempt returned CVVDP numbers identical to the
   local MPS run, but the libjxl Butteraugli/SSIMULACRA2 pair was unusable
   (e.g. `butteraugli` 437 and `ssimulacra2` NaN on a stream whose local pair
   is 2.17/83.24). A NaN SSIMULACRA2 is clamped to a perfect score by
   `tools/kpi.py`, which silently inflates K. Shipping that path would measure
   a different metric stack, not the gate.

Remote scoring is therefore a separate task: pin a Linux decoder, pin the
metric tools, and pass an agreement gate on the five pins before it can enter
the loop. `--score remote` and `--score auto` exit 2 with this explanation.

## What runs where

| stage | local envelope | Modal envelope |
|---|---|---|
| source | `build-release/av2enc` in the worktree | the worktree is mounted and built in the image (`release` preset) |
| encode | one process per job, `--cpu-threads 8` | one container per `(arm, QP)`, `cpu=1`, `memory=200 MiB`, `--cpu-threads 1`, up to 24 at once |
| bracket | secant step, measured bracket | identical: same `rate_kbps`/`next_qp` decisions from the same byte counts |
| score | `tools/oracle.py` (AVM `avmdec`, CVVDP on MPS) | the same `tools/oracle.py` on the returned IVF bytes |
| ceiling | local oracle | local oracle |

The QP scheduler mirrors `envelope.bracket_arm` exactly. Each round every open
arm proposes one QP; the round goes out as one Modal batch; the returned byte
counts drive the next proposal. On the validation spec the remote scheduler
walked the same QP sequence as the local tool, and every returned IVF was
byte-identical to the local 8-thread release build.

## Env-only gates

Every gate added by the current RD lanes is an `AV2ENC_*` environment switch,
not a flag. The envelope spec already carries per-arm `env`; the wrapper
merges `--base-env` into every arm exactly like `tools/envelope.py` and sends
the merged env with each encode job. `tools/modal_rd.py --mode matrix` accepts
`env` per entry:

```json
[
  {"name": "cand", "qps": [106, 108],
   "flags": ["--video", "--frames", "60", "--fast-decode"],
   "env": {"AV2ENC_RD_UVFULL": "1", "AV2ENC_TOOL": "1"},
   "clip": "240p", "frames": 60}
]
```

`--env '{"AV2ENC_...": "1"}'` applies one JSON object to every job.
`tools/modal/rd_encode.py` has the same `--env` option.

## Worktree mount

The envelope runs while the lane's patch is uncommitted, so the wrapper mounts
the worktree itself, not a committed tree. `tools/envelope_modal.py
--worktree <lane>` snapshots that tree (build products, media and `.git`
excluded) into the image and builds `release`. The spec's `bin` path is used
by the local envelope only; on Modal it maps to `/src/build-release/av2enc`.

`tools/modal_rd.py --repo <lane>` is the same mount for the matrix mode. Modal
builds the image before the entrypoint parses arguments, so `--repo` is also
scanned from `sys.argv` at import; `AV2ENC_MODAL_REPO` works too. The
entrypoint checks that the mount matches.

## Worker shape

- **encode:** `cpu=1`, `memory=200`, `timeout=1800`, `--cpu-threads 1`,
  `max_containers=24`. Measured peak RSS is 50.5 MiB at 240p and 245.5 MiB at
  720p, so 200 MiB is comfortable for the 240p lanes. The 720p pin does not
  fit: `tools/modal_rd.py` routes it to `encode_large` (`memory=512`).
  `AV2ENC_MODAL_MAX_CONTAINERS` overrides the encode pool.

The x86 CPU build uses the scalar reference primitives (the Mac uses NEON), so
a 240p 60-frame encode at 1 thread is minutes, not seconds. One core per job
keeps each job matched to its reservation and lets 24 jobs share the pool.

## Commands

Pin check on Linux (release build, 1 thread):

```sh
python3 -m modal run tools/modal_rd.py --mode check
```

A worktree matrix with env gates:

```sh
python3 -m modal run tools/modal_rd.py --mode matrix --matrix jobs.json \
    --repo ~/work/rd-loop-1
```

The envelope, same arguments as `tools/envelope.py`:

```sh
python3 tools/envelope_modal.py --spec build/agent-loop/spec.json \
    --worktree ~/work/rd-loop-1 --threshold 0.15 --min-arms 3 \
    --base-env '{"AV2ENC_PIPE_EXACT":"1"}' --json /tmp/envelope.json
```

The ten lanes with the Modal measurement (default stays the local envelope):

```sh
AV2ENC_ENVELOPE=$PWD/tools/envelope_modal.py \
AV2ENC_LANES=10 AV2ENC_ITERATIONS=10 tools/agent_loop_multi.sh
```

`tools/agent_loop.py --envelope-tool tools/envelope_modal.py` is the
per-runner equivalent; use an absolute path because the agent loop runs with
the lane worktree as its working directory. `tools/agent_loop_multi.sh` keeps
the base env and the ceiling flags per lane, so the wrapper receives the same
gate configuration as the local tool.

Only the worktree mount and the release build rerun per envelope; the Modal
image layers cache by content hash. A lane patch changes the source, so
expect a ~4 min image rebuild before the first encode of each iteration.

## Validation

Same worktree, same spec (2 arms), same `--base-env`; the local tool ran the
8-thread release build and the Modal wrapper the 1-thread container build.
The bracket walked QPs 106, 112, 113, 118, 119, 120 and both tools produced
the same brackets. Every Modal stream was byte-identical to its local
counterpart.

| arm | local K @100 | Modal K @100 | delta |
|---|---:|---:|---:|
| base | 86.0132 | 86.0132 | +0.0000 |
| cand | 85.1803 | 85.1803 | +0.0000 |

Every stream is byte-identical: 9/9 encodes matched the local 8-thread
release build at the same QP (`base` 106/112/118/119, `cand`
106/113/116/118/120), and both tools walked the same QP sequence. The K delta
is exactly zero because the local oracle scored the same bytes; the only
changed variable is the encode host and thread count.

Wall clock on this Mac (the ten-lane loop was also running, so the local
CVVDP was contended):

| path | wall | encode | score | notes |
|---|---:|---:|---:|---|
| `tools/envelope.py` | 953 s | ~9 local encodes | ~local CVVDP | 15:53 |
| `tools/envelope_modal.py` | 1122 s | 827 s (9 jobs) | 76 s | 18:42; ~4 min image/startup |

The Modal path is not faster for one lane on a quiet Mac. It wins when the ten
lanes run together: the encodes use a 24-container pool instead of one process,
and the local CVVDP work is limited to the oracle calls the gate needs.

Per-encode wall at 1 thread: 68-216 s (median ~120 s) for the 240p 60-frame
spec; the spread is cloud-vCPU contention. Peak encode concurrency for this
2-arm spec is 2.

A 12-job matrix (2 configs x 6 QPs, one env-only gate per config) measured
peak concurrency **12** of the 24-container cap and **205.5 s** of encode
wall (mean 121.6 s/job, range 67.4-201.7 s; total run 9:26 including a cold
image rebuild). Every 240p job passed in the 200 MiB container, and the two
envs produced different streams at every QP, so the per-entry `env` is live.

Pin check: **PASS**, all five pins byte-identical on the Linux release build:
`default` 25,009 B `a1d0dba1`, `quality` 25,141 B `3984677b`, `nightly`
23,690 B `1b99127a`, `explambda` 24,113 B `7f03eb6e`, `720p` 36,572 B
`c4ba2813`. The cold check was 23.5 min (4 min image build + 18.9 min encode,
peak 5); the 720p pin at 1 thread dominates. The 240p-only pins are ~2-3 min
each once the image is cached.

## Remaining bottleneck

The encodes are off the Mac, but the gate still scores two streams per arm
with the local AVM oracle and CVVDP: ~30-60 s per stream, serialised behind
one GPU/MPS device. That is the ten-lane measurement's remaining local cost;
the encode half no longer competes for it.

Enabling remote scoring later (do not use it before the parity gate):

```sh
# Build the metrics image (pinned libjxl + dav2d + cvvdp 0.5.7):
WITH_METRICS=1 python3 -m modal run tools/modal_rd.py --mode check --with-metrics
# or, in a plain container:
WITH_METRICS=1 sh tools/modal_image_setup.sh
```

A future wrapper must first show that the container's per-point JOD /
Butteraugli / SSIMULACRA2 match the local oracle on the five pins, with a
pinned Linux decoder. Until then the encode-only split is the shipped path.
