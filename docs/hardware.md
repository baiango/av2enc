# Hardware identity

**Primary target: 8 GB Ampere (`sm_86`).** Target validation requires an RTX
3060 that **enumerates 8192 MiB** in `nvidia-smi`. The development box is a
12 GB RTX 3060 running the `compat8g` 7,500 MB cap; it is a compatibility proxy
only, not a performance/QBS validation host. Its 192-bit/360 GB/s memory is
faster than the 8 GB card's 128-bit/240 GB/s.

CSS/CVVDP speed scores are hardware-specific. Every `quality.py` report records
the CPU model, physical/logical cores, RAM, OS, and GPU. `score_speed.py` refuses
to compare reports from different CPU models unless `--allow-mixed-hardware` is
passed.

## Mac benchmark host (AVM/CVVDP only)

**Not an encoder build target.** This machine is used only to run AVM and score
quality with CVVDP/MPS.

| | |
|---|---|
| CPU | **Apple M4**, 10 cores |
| RAM | 16 GiB (reported 17.2 GB) |
| OS | macOS 26.6.2 (arm64) |
| Hostname | Mac-mini.local |
| GPU | Apple M4 integrated (no CUDA) |
| CVVDP device | MPS |
| Encoder | AVM macOS arm64 nightly, commit `cd7d8bc3db` |

This is the machine used for the AVM baseline. It is fast single-threaded, which
matters for encoder orchestration, mode decision, and entropy coding.

## Vast.ai box (8 GB emulation / dev)

| | |
|---|---|
| Role | Dev box; must run `compat8g` (7,500 MB) to emulate 8 GB Ampere |
| CPU | **2× Intel Xeon E5-2690 v4 @ 2.60 GHz** |
| Topology | 14 cores/socket, 2 threads/core, 2 sockets = **28C / 56T** |
| Observed scaling | `CPU(s) scaling MHz: 38–43%` (effective clock is well below 2.6 GHz; varies by run) |
| RAM | 125 GiB |
| GPU | **RTX 3060 12 GB**, driver 570.181, CUDA 12.8, sm_86 |
| PCIe | Gen3 ×16 |
| Disk | 16 GiB overlay, **not persistent** |
| OS | Ubuntu, unprivileged container |

The E5-2690 v4 is a 2016 Haswell-EP part. A modern desktop/Mac CPU will beat it
substantially in single-thread performance, and the container often runs it at a
low scaling frequency. This matters for the CPU-side encoder pipeline (entropy
coding, bitstream writing, rate control).

## Rule

Do not compare CSS across the two machines. For a like-for-like speed ratio:

1. Run AVM on the **same** machine as our encoder.
2. Use the same source clip, resolution, frame rate, and decoder.
3. Compare at iso-CVVDP, and report the CPU model with the score.

## Rental proxies

| Card | VRAM | Role | Caveat |
|---|---:|---|---|
| RTX 3060 12 GB | 12 GB | cheap dev box, `compat8g` cap | same GPU as 8 GB but 192-bit/360 GB/s vs 128-bit/240 GB/s; speed is optimistic |
| RTX 3060 Ti 8 GB | 8 GB | capacity validation only | 4864 CUDA cores, 256-bit/448 GB/s; much faster than either 3060, not a speed proxy |
| RTX 3060 8 GB | 8 GB | actual target | preferred when available |

Do not pay a premium for a 3060 Ti just to get 8 GB. Use the cheaper 3060 12 GB
with the 7,500 MB cap for development, and rent an 8 GB card only for a short
capacity/OOM validation run. The Ti is a different performance class.

## Capture

```sh
# Linux
lscpu | grep -E 'Model name|Socket|Core|Thread|scaling'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# macOS
sysctl -n machdep.cpu.brand_string
sysctl -n hw.physicalcpu hw.logicalcpu hw.memsize
```

`python3 tools/quality.py ... --json out.json` captures the same fields
automatically under `host`.
