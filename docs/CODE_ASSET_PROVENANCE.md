# Code and asset provenance

GraphLease is an independent repository and produces new workloads and results.

The fixed Qwen3-1.7B file manifest and prior A100/910B asset qualification were
established in the user's PolicyMesh/UpdatePassport workspaces. GraphLease reuses
only the identical public model revision and deployed path; it does not reuse
their scientific outputs or Gate decisions.

The scoped vLLM worker-dispatch observation pattern is adapted from the user's
KneeTP `run_offline_probe.py` at commit
`d8b2269` ancestry. GraphLease uses it only in a separate instrumented replay.
Formal timing has no dispatch hook.
