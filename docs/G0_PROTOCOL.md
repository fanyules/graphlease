# GraphLease Gate 0: graph portfolio lifecycle qualification

## Question and scope

Gate 0 asks whether resident graph selection is a real cross-runtime resource
lifecycle problem. It does not implement GraphLease, a router, a new kernel, or
a runtime patch.

The mechanism under test is:

```text
workload phase changes
-> useful capture-size distribution changes
-> resident graph coverage trades padding/fallback against graph/KV/stream cost
-> replacing a portfolio creates a drain-to-ready service gap
-> a warm heterogeneous island may temporarily absorb that gap
```

The fixed platforms are one Qwen3-1.7B BF16 TP1 replica on A100 and one on
Ascend 910B. Each platform uses its existing vLLM 0.23 stack. The 1 GbE link is
not a token-level data path.

The Gate is motivated by two primary implementation sources. NVIDIA reports
that denser CUDA graph capture sets can reduce padding but consume graph memory,
reduce KV capacity, and extend startup time. vLLM-Ascend documents that capture
sizes are bucketed, effective modes may be normalized, and PIECEWISE ACL Graph
coverage can exhaust stream resources:

- <https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.html>
- <https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/developer_guide/Design_Documents/ACL_Graph.md>

Those public measurements establish plausibility, not a result for this model or
these devices.

## Frozen engine and workload

The exact model revision, engine limits, seeds, phase traces, capture sizes, and
thresholds are in `configs/g0.json`.

Two deterministic closed-loop service phases are replayed:

- `low_short`: 64-token prompts, 32 generated tokens, concurrency concentrated
  between 3 and 16;
- `high_long`: 256-token prompts, 128 generated tokens, concurrency concentrated
  between 33 and 123.

The concurrency lists are fixed irregular traces rather than bucket-adversarial
search results. Every wave enqueues its requests atomically and drains fully
before the next wave. Greedy decoding ignores EOS so all correct plans execute
the same generated-token work.

Each formal portfolio runs in three fresh processes. Phase order alternates by
restart. Each phase has one untimed warm-up wave, one uninstrumented formal
trace, and one separate instrumented replay. Profiler or dispatch hooks never
contribute timing evidence.

The timing driver submits pre-tokenized requests through public
`LLMEngine.add_request()` with cumulative output enabled. `LLM.enqueue()` is not
used because vLLM 0.23 intentionally converts that offline convenience path to
final-only output, which cannot expose TTFT. On A100, the frozen environment
handoff's existing `qwen36-shard/bin` PATH prefix supplies `ninja` for FlashInfer
JIT; the Python interpreter remains `rimlink-vllm023`.

## Portfolios

Four portfolios enter service comparison:

1. `eager`: no graph execution;
2. `default`: the platform's unmodified effective vLLM graph configuration;
3. `small_dense`: exact coverage for sizes 1--32 plus sparse large anchors;
4. `large_dense`: the default small core plus four-token spacing from 20--128.

`small_dense` and `large_dense` contain 38 and 37 requested sizes respectively,
so a positive result cannot be explained merely by giving one specialized plan
many more graphs. Neither deletes common small anchors solely to create a cliff.

`coverage_union` is the exact union of the two specialized sets. It performs
initialization and resource measurement only; it never enters performance
selection. If the union is cheap and captures successfully, graph scarcity is
not established and the project stops.

## Measurements

### Formal service measurements

For every platform, portfolio, restart, and phase, record:

- useful output tokens per second; a token is useful only after within-platform
  exact output equality to the same-restart eager reference passes;
- request TTFT, TPOT, end-to-end latency, and p99;
- step latency and total phase time;
- failed, OOM, timeout, and preemption status;
- output token digest.

### Separate instrumented replay

Record effective requested and realized capture sizes, FULL/PIECEWISE/NONE
dispatch counts, padded and unpadded token counts, and scheduler preemption
fields. The worker hook is removed after the instrumented replay.

### Resource and lifecycle measurements

Record process-to-ready initialization time, shutdown time, device allocated and
reserved HBM after initialization, effective KV block count and token capacity,
and realized graph count.

CUDA graph HBM is inferred only by paired resident-memory and KV-capacity deltas
against eager/default under the same `gpu_memory_utilization`; it is not inferred
from one absolute allocator value.

The current CANN stack exposes graph objects and confirmed stream-exhaustion
errors but not a stable public API for an exact device-wide stream counter.
Therefore the registered ACL stream evidence is:

- requested versus effective capture sizes;
- captured ACL graph-entry count and vLLM compilation counter;
- the runtime's confirmed stream-resource exhaustion signature, if present.

These are reported as a stream-footprint proxy, never as an invented exact
stream count.

## G0-P: mandatory plan-invariance preflight

Before the formal matrix, each platform runs eager, default, small-dense, and
large-dense on fixed 7-request and 63-request cases. Continue only if:

- every non-eager output digest exactly equals eager within the same platform;
- the requested/effective plan is recorded;
- dispatch/padding statistics are observed;
- KV token capacity and initialization resource measurements are observed.

Any 910B plan-invariance failure stops the CUDA/CANN claim immediately. Timing
from G0-P is diagnostic only.

## Four Gate questions

### Q1. Phase-dependent optimum

On each platform, `low_short` and `high_long` must select different winning
resident graph portfolios. A winner must beat the runner-up graph portfolio
in at least two of three paired restarts and satisfy either:

- paired-bootstrap 95% lower bound of correct-token goodput gain at least 5%; or
- paired-bootstrap 95% lower bound of p99 reduction at least 10%.

The result must hold on both platforms.

### Q2. Real graph scarcity

On each platform, `coverage_union` versus default must cause at least one of:

- KV token-capacity loss of at least 3%;
- resident-memory increase of at least 512 MiB;
- on 910B, requested-size filtering or a confirmed stream-resource capture
  failure.

If all useful sizes initialize without those costs, portfolio scarcity is absent
and GraphLease stops.

### Q3. Static-versus-oracle opportunity

The per-phase oracle chooses the fastest correct portfolio independently for
each phase. Static candidates are eager, default, small-dense, and large-dense.
The two phases receive equal weight by normalizing each static candidate to the
phase oracle before aggregation.

On both platforms, the 95% lower bound must show either:

- at least 10% correct-token-goodput opportunity over the best static plan; or
- at least 15% p99-latency opportunity.

A frozen black-box latency baseline trains only on restarts 0 and 1 and is
evaluated on held-out restart 2. If its oracle regret is at most 5%, an explicit
graph contract is unnecessary and the project stops.

### Q4. Maskable lifecycle gap

Run only if Q1--Q3 pass. Measure `drain complete -> replacement ready` for the
winning phase transition. A warm-island replay uses an arrival rate fixed by the
registered rule: 70% of the slower island's correct-token phase capacity. It
must finish without new preemption or backlog and with p99 inflation at most
10% for at least the measured gap duration.

This is a capacity-and-gap qualification, not a GraphLease controller.

## Decision

- All four questions true: implement GraphLease's resident-portfolio contract,
  staggered lifecycle, and TP-consistent dispatch.
- Only a static portfolio improves results: stop at a tuning result with major
  overlap risk.
- Benefits require artificial removal of common buckets: stop.
- Useful union coverage is cheap: stop because scarcity is absent.
- Held-out black-box latency selection is within 5% of oracle: stop.
- 910B TP1 is not plan-invariant: stop the cross CUDA/CANN claim.

No threshold, phase trace, capture-size set, or baseline may change after the
first accelerator output.
