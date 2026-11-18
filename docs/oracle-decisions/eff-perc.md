# eff-perc — oracle decision log

- status: **open**
- source: `{"commit":"0bf1d001","kind":"binary","note":"Main build-release contains the env-gated eff-perc core."}`
- gate: **open** — ceiling 90.116 >= SVT 88.8
- gate ceiling: **90.116 K** (ceiling 90.116 >= SVT 88.8)
- measured max ceiling: **90.165 K** (SVT gap 1.365)
- envelope: **85.099 K** from full (lambda=0.18)
- full ladders: 2 of 5 arms (saved 3)
- verdict: **NO-GO: below SVT 88.8**

Extended perceptual RD core (AV2ENC_EFF_PERC=1) on the drift recipe. Inner sweep: lambda scale. Known best lambda 0.18 at 85.10 matched K.

Sweep: skipped — full envelope 85.099 < SVT 88.8; no full sweep

| step | arm | qps | K | kbps | decision | reason |
|---|---|---|---:|---:|---|---|
| gate-probe | lambda=0.18 | 119,120 | 85.099 | 100.000 | measure-ceiling | probe with source replacement |
| gate-probe | lambda=0.3 | 118,119,120,121 | 84.740 | 99.056 | invalid-probe | probe did not bracket the target rate |
| screen | lambda=0.15 | 119,120 | 84.858 | 100.000 | screen | screen without segments |
| screen | lambda=0.2 | 118,119,120 | 84.801 | 100.000 | screen | screen without segments (QP set extended to bracket 100 kbps) |
| screen | lambda=0.25 | 118,119,120 | 84.968 | 100.000 | screen | screen without segments (QP set extended to bracket 100 kbps) |
| promote | lambda=0.18 | 119,120 | 85.099 | 100.000 | promote | screen best 85.099; margin 0.50 keeps 4; finalist budget 2 |
| full-ladder | lambda=0.18 | 118,119,120,121 | 85.099 | 100.000 | measure-full | full QP ladder with source segments |
| full-ladder | lambda=0.25 | 118,119,120,121 | 84.968 | 100.000 | measure-full | full QP ladder with source segments |
| decide | lambda=0.18 | 118,119,120,121 | 85.099 | 100.000 | no-go | full envelope 85.099 < SVT 88.8; stop, no full sweep |

Surrogate: `gp-rbf` with EI acquisition.

| arm | mean | sigma | EI | UCB |
|---|---:|---:|---:|---:|
| lambda=0.15 | 84.860 | 0.004 | 0.000 | 84.867 |
| lambda=0.18 | 85.095 | 0.004 | 0.000 | 85.103 |
| lambda=0.2 | 84.804 | 0.004 | 0.000 | 84.811 |
| lambda=0.25 | 84.967 | 0.004 | 0.000 | 84.975 |
| lambda=0.3 | 85.137 | 0.102 | 0.056 | 85.341 |

