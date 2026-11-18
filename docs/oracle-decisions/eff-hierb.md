# eff-hierb — oracle decision log

- status: **open**
- source: `{"binary":"build-release/av2enc","commit":"fd59a737","kind":"worktree","note":"Prebuilt exp/eff-hierb binary; read-only use.","worktree":"../eff-hierb"}`
- gate: **open** — ceiling 89.201 >= SVT 88.8
- gate ceiling: **89.201 K** (ceiling 89.201 >= SVT 88.8)
- measured max ceiling: **89.201 K** (SVT gap 0.401)
- envelope: **84.412 K** from full (default)
- full ladders: 1 of 1 arms (saved 0)
- verdict: **NO-GO: below SVT 88.8**

Hierarchical B with the merged full tool set, BA refresh allocation and two-sided AVERAGE compound. Reference envelope 84.4 K from docs/eff-hierb.md. Uses the existing worktree binary; the tool never builds in or modifies that worktree.

Sweep: skipped — full envelope 84.412 < SVT 88.8; no full sweep

| step | arm | qps | K | kbps | decision | reason |
|---|---|---|---:|---:|---|---|
| gate-probe | default | 117,118 | 84.412 | 100.000 | measure-ceiling | probe with source replacement |
| promote | default | 117,118 | 84.412 | 100.000 | promote | screen best 84.412; margin 0.50 keeps 1; finalist budget 1 |
| full-ladder | default | 117,118,119 | 84.412 | 100.000 | measure-full | full QP ladder with source segments |
| decide | default | 117,118,119 | 84.412 | 100.000 | no-go | full envelope 84.412 < SVT 88.8; stop, no full sweep |

Surrogate: `fallback` with EI acquisition.

| arm | mean | sigma | EI | UCB |
|---|---:|---:|---:|---:|
| default | 84.412 | — | — | — |

