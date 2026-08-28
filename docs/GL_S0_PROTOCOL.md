# GL-S0: Qwen3-14B TP4 graph-entry scale audit

Status: **prospective, resource-only scale-transfer Gate**

GL-S0 does not reopen or reinterpret the failed GraphLease G0-M Gate. The
negative G0-M decision and repository state at commit
`93435649eedcacc14130b445157a26a0c1b4e891` remain frozen. GL-S0 tests a new,
narrow hypothesis: whether the same logical graph portfolio has a
scale-dependent and runtime-dependent physical resource cost at Qwen3-14B TP4.

## Motivation fixed before output

The retained Qwen3-1.7B 910B measurements contain 30 ACL wrappers. Runtime
default captured 19 sizes and materialized 570 graph entries; the frozen
60-size coverage union materialized 1,796 entries. These are direct raw-result
counts, not estimates from latency. The 14B probe is a held-out scale transfer:
it measures the wrapper and entry counts rather than assuming the old
approximately 1,800-entry implementation budget is a universal hard limit.

## Frozen matrix

| Axis | Values |
| --- | --- |
| Model | Qwen3-14B BF16 |
| Parallelism | TP4 |
| Platforms | 4 x A100 PCIe; 4 x Ascend 910B |
| Portfolios | runtime default; the original 60-size `coverage_union` |
| Restarts | 3 sequential fresh processes per platform and portfolio |

Engine limits remain those of the original GraphLease Gate:
`max_model_len=768`, `max_num_seqs=128`,
`max_num_batched_tokens=4096`, and `gpu_memory_utilization=0.8`. Prefix caching,
async scheduling, and EngineCore multiprocessing remain disabled. The 910B
runtime keeps `enable_cpu_binding=true` and `enable_flashcomm1=false`.

The probe initializes the engine, reads each rank's resources, and shuts the
engine down. It issues zero requests. It does not run phases, Q1/Q3, a router,
network traffic, or a controller; it does not modify a kernel, merge, runtime,
or capture quantum.

## Required evidence

Every run records:

- requested and resolved capture sizes;
- TP rank count, ACL wrapper and concrete graph-entry count when available;
- compilation counters, including captured graph count;
- per-rank allocated, reserved, free, and total device memory;
- KV block count and token capacity;
- model-init and process-to-ready time;
- filtering or a retained initialization error and traceback.

The A100 and 910B model `config.json` must match the frozen SHA-256. A successful
run must return four rank snapshots. Formal resource comparisons use the
immediate post-initialization snapshot; no materialization workload is added.

## Fixed adjudication

All three runtime-default runs on both platforms must initialize. Otherwise the
Gate is technically invalid and is not positive.

1. **Cross-runtime resource constraint.** Continue only if all three A100 union
   runs initialize with all 60 requested sizes, while at least two of three
   910B union runs either filter the requested set or fail with a registered
   stream-resource marker. This motivates a separate GraphBudget direction,
   not restoration of GraphLease.
2. **Scale stress on both runtimes.** If all union runs initialize, compute
   paired union-minus-default resource changes by restart. A platform is
   stressed when its median KV-capacity loss is at least 3%, or its median
   per-rank reserved-memory increase is at least 512 MiB. Both platforms must
   be stressed to justify a new scale-conditioned phase Gate.
3. **Startup-only or cheap complete capture.** If both platforms fully capture
   the union without the two-platform capacity criterion, stop the graph
   portfolio family. A median process-to-ready increase of at least 10% is
   reported as startup-only and is not a positive mechanism result.
4. **Other union failures or one-sided memory pressure.** Stop without a
   positive claim. A100-only exhaustion, non-resource runtime errors, missing
   results, and a one-platform capacity threshold do not satisfy either
   continuation condition.

No additional restart, portfolio, capture-size search, workload, or model is
allowed after seeing GL-S0 output.

## Scope of any positive result

A positive GL-S0 result establishes only graph-resource scale or runtime
asymmetry. It does not establish phase-dependent portfolio choice, an oracle
gap over a strong static configuration, failure of a black-box latency router,
or amortization of drain/capture/rejoin. Those require a separately frozen
future Gate. The known Qwen3-14B TP4 output instability on 910B remains a
blocker for such a performance Gate, but it does not affect this zero-request
resource audit.
