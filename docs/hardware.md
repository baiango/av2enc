# Hardware identity

CSS/CVVDP speed scores are hardware-specific. Every `quality.py` report records
the CPU model, physical/logical cores, RAM, OS, and GPU. `score_speed.py` refuses
to compare reports from different CPU models unless `--allow-mixed-hardware` is
passed.

## Mac benchmark host (AVM baseline)

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

## Vast.ai box (GPU development)

| | |
|---|---|
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
