# nightly — oracle decision log

- status: **open**
- source: `{"commit":"0bf1d001","kind":"binary","note":"The main build-release binary passed with --bin."}`
- gate: **closed** — ceiling 87.959 < SVT 88.8
- gate ceiling: **87.959 K** (ceiling 87.959 < SVT 88.8)
- measured max ceiling: **87.959 K** (SVT gap -0.841)
- envelope: **82.048 K** from screen (default)
- full ladders: 0 of 1 arms (saved 1)
- verdict: **gate closed below SVT 88.8**

Flat key+P nightly baseline. No inner knobs; one architecture point. The source-segment ceiling is expected to close the gate below 88.8.

Sweep: skipped — gate closed below the SVT wall; no full sweep

| step | arm | qps | K | kbps | decision | reason |
|---|---|---|---:|---:|---|---|
| gate-probe | default | 105,106,107 | 82.048 | 100.000 | measure-ceiling | probe with source replacement (QP set extended to bracket 100 kbps) |
| gate | default | 105,106,107 | 82.048 | 100.000 | close | ceiling 87.959 < SVT 88.8 |

