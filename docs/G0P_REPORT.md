# GraphLease G0-P report

Status: **PASS — formal matrix unblocked**

Executed repository: `4a9cd771efbba342110ebbea0b3ff9b66c0dc803`

## Decision

Qwen3-1.7B BF16 TP1 is exactly plan-invariant for both registered preflight
cases on A100 and Ascend 910B. All requested portfolios initialized, their
effective capture sets and dispatch paths were observable, and each run exposed
a nonzero KV token capacity. The frozen adjudicator returned
`pass_unblock_formal`.

No preflight latency enters the formal comparison.

## Valid rows

| Platform | Portfolio | Effective capture sizes | KV token capacity | Process-to-ready | Status |
|---|---|---:|---:|---:|---|
| A100 | eager | 0 | 262,704 | 16.90 s | pass |
| A100 | default | 35 | 258,928 | 55.62 s | pass |
| A100 | small-dense | 38 | 258,928 | 56.54 s | pass |
| A100 | large-dense | 37 | 258,928 | 58.11 s | pass |
| 910B | eager | 0 | 422,912 | 37.34 s | pass |
| 910B | default | 19 | 423,296 | 64.36 s | pass |
| 910B | small-dense | 38 | 423,168 | 70.37 s | pass |
| 910B | large-dense | 37 | 423,168 | 71.75 s | pass |

For every platform and portfolio, the 7-request and 63-request output digests
matched the same-platform eager digests exactly. Instrumented dispatch rows were
present. Small-dense eliminated decode padding in the 7-request case; default
and large-dense padded that case to the next supported size. These observations
are mechanism diagnostics, not formal performance results.

## Retained construction failures

Four failed engineering artifacts are retained under `results/g0/attempts/`:

- missing `ninja` on the initial A100 PATH;
- final-only output from the first 910B offline convenience path;
- caller-ID versus internal suffixed-ID mapping on one A100 and one 910B run.

The repairs are documented in `docs/G0_RUNNER_REPAIR.md`. They changed no
workload, graph set, threshold, model, device, or runtime plan. The successful
rows all postdate the final canonical freeze.

## Next boundary

G0-M may run the frozen four-portfolio, two-phase, three-fresh-process matrix and
the resource-only coverage union. G0-L and any GraphLease controller remain
blocked.
