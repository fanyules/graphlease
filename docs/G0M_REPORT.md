# GraphLease G0-M report

Status: **FAILED — GraphLease stopped**

Executed runner/freeze commit: `4a9cd771efbba342110ebbea0b3ff9b66c0dc803`

Machine verdict: `stop_no_stable_phase_dependent_optimum`

## Decision

The complete A100 and 910B matrix passed execution, exact within-platform output
equality, instrumented/formal output equality, and dispatch observability. It did
not establish a graph portfolio lifecycle problem. G0-L is not run and no
GraphLease controller is implemented.

The primary registered failure is Q1. Q2 and Q3 independently point to the same
stop decision: useful union coverage was inexpensive on this model, the best
static graph portfolio was already close to the per-phase oracle, and a simple
held-out black-box phase mapping was within 5% of oracle on both platforms.

## Q1: no stable cross-platform phase-dependent optimum

| Platform | Phase | Median winner | Runner-up | Paired wins | Goodput gain L95 | p99 reduction L95 | Stable? |
|---|---|---|---|---:|---:|---:|---|
| A100 | low-short | small-dense | large-dense | 2/3 | -5.66% | -8.01% | no |
| A100 | high-long | default | large-dense | 3/3 | 0.14% | -0.20% | no |
| 910B | low-short | default | small-dense | 2/3 | -0.73% | 0.40% | no |
| 910B | high-long | default | small-dense | 2/3 | -1.45% | -2.19% | no |

A100's median label changed between phases, but the low-short effect changed sign
across restarts and failed both registered confidence thresholds. On 910B,
default was the median winner in both phases. Therefore there is neither a
stable two-phase flip on A100 nor any flip on 910B.

Median correct-token goodput among graph portfolios was tightly clustered:

| Platform | Phase | Default | Small-dense | Large-dense |
|---|---|---:|---:|---:|
| A100 | low-short | 1,712.7 | 1,752.4 | 1,750.7 |
| A100 | high-long | 6,900.4 | 6,807.8 | 6,867.6 |
| 910B | low-short | 887.5 | 885.5 | 874.3 |
| 910B | high-long | 3,434.3 | 3,410.8 | 3,393.0 |

## Q2: portfolio scarcity not established

The 60-size coverage union initialized successfully on both platforms. Neither
runtime filtered a requested size and 910B reported no confirmed stream-resource
failure.

| Platform | Default KV tokens | Union KV tokens | KV loss | Resident-memory delta | Result |
|---|---:|---:|---:|---:|---|
| A100 | 262,848 | 258,928 | 1.49% | -448 MiB | below threshold |
| 910B | 423,168 | 423,168 | 0% | +2 MiB | below threshold |

The A100 allocator delta is negative because the automatic KV allocation and
reserved-memory boundary move together; it is not interpreted as negative graph
memory. The directly comparable KV loss is still below the registered 3%
scarcity threshold. On 910B, the union added no measurable KV loss and only a
2 MiB reserved-memory delta. For Qwen3-1.7B TP1, useful coverage is therefore
cheap enough that graph scarcity is not demonstrated.

## Q3: static and black-box baselines close the opportunity

| Platform | Best static | Goodput-gap L95 | p99-gap L95 | Held-out black-box regret |
|---|---|---:|---:|---:|
| A100 | large-dense | 0.071% | 0% | 3.66% |
| 910B | default | 0% | 0% | 0.21% |

Neither platform approaches the registered 10% goodput or 15% p99 opportunity.
The frozen black-box mapping is also within 5% of oracle on both platforms,
triggering an additional stop condition.

## Lifecycle observation

Graph initialization is visibly nonzero: preflight process-to-ready time rose
from 16.9 s eager to about 55.6--58.1 s for graph plans on A100, and from 37.3 s
to about 64.4--71.8 s on 910B. This is a real restart gap, but Q4 is conditional
on Q1--Q3. A costly gap alone does not justify rotating portfolios when the
portfolio optimum does not change stably and static/black-box choices already
close the oracle gap.

## Final interpretation

This Gate does not claim graph capture is free in general. It establishes the
tested boundary: Qwen3-1.7B BF16 TP1 on these A100/vLLM and 910B/vLLM-Ascend
stacks has graph-versus-eager benefits and measurable startup cost, but no useful
cross-runtime portfolio scarcity or stable dynamic portfolio opportunity.

The correct project decision is therefore:

- stop GraphLease;
- do not implement staggered drain/recapture/rejoin;
- do not add a controller, RTX, larger model, or new portfolio search to rescue
  the result;
- retain the data as a negative systems boundary, not a B-class tuning paper.

## Evidence

- machine decision: `results/g0/G0M_DECISION.json`;
- 24 formal fresh-process runs: `results/g0/formal/`;
- two union resource probes: `results/g0/resource/`;
- G0-P evidence and retained construction failures: `results/g0/preflight/` and
  `results/g0/attempts/`.

G0-L was not run, as registered.
