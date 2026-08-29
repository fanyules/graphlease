# GB-Q0: official-stack qualification

Status: **prospective; no GB-Q0 output exists**

GB-Q0 asks one question: does the Qwen3-14B 60-size graph portfolio still hit
the ACL resource cliff after replacing the old HDK 25.2.3 release-candidate
stack with the current officially matched stack?

The frozen GL-S0 decision is immutable. GB-Q0 has a new classifier and cannot
rewrite GL-S0.

## Required stack

The complete compatibility row is frozen as:

| Component | Required value |
| --- | --- |
| Ascend driver / HDK | 25.5.1 |
| Firmware | 7.8.0.6.201, package 25.5.1 |
| CANN / NNAL | 9.1.0 |
| PyTorch / torch-npu | 2.10.0 / 2.10.0.post4 |
| vLLM / vLLM-Ascend | 0.23.0 / 0.23.0 |
| Official image | `quay.io/ascend/vllm-ascend:v0.23.0` arm64 |

Every process must record and validate all seven version boundaries before
engine initialization. A release-candidate image, a mixed compatibility row,
or HDK below 25.5.1 makes the Gate technically invalid.

The compatibility row is taken from the official vLLM-Ascend versioning
matrix, and the stream-budget requirement from its official release notes:

- <https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html>
- <https://docs.vllm.ai/projects/ascend/en/main/user_guide/release_notes.html>

## Frozen workload and order

GB-Q0 uses Qwen3-14B BF16 TP1 and issues zero inference requests. It preserves
the GraphLease engine limits: `max_model_len=768`, `max_num_seqs=128`,
`max_num_batched_tokens=4096`, and `gpu_memory_utilization=0.8`.

There are three trials on physical NPUs 0, 1, and 2. Each trial is sequential:

```text
fresh default_pre process
fresh U60 process
fresh default_recovery process only if U60 failed
```

The default recovery uses the same physical NPU as the failed U60 process. It
tests cleanup and order effects; it is not an additional portfolio search. No
other capture set, TP degree, model, workload, router, or performance timing is
allowed.

Recovery requires the same effective default sizes, ACL wrapper count, and ACL
entry count as the pre-U60 default. Its KV token capacity may be at most 1%
lower, allowing bounded fresh-process memory-profiling variation.

## Recorded evidence

Each engine process records:

- driver, firmware, CANN, torch, torch-npu, vLLM, and vLLM-Ascend versions;
- model-config hash and repository/freeze hashes;
- requested and resolved capture sizes;
- ACL wrapper and concrete entry counts;
- compilation counters;
- per-rank allocated, reserved, free, and total HBM;
- KV block count and token capacity;
- process-to-ready time as a diagnostic only;
- complete initialization exception and traceback.

## Resource classifier

A U60 failure is a registered resource cliff only when one error record contains
all five strings:

1. `AclmdlRICaptureBegin`;
2. `207005`;
3. `Alloc sq cq fail`;
4. `0x7020023`;
5. `resource alloc fail`.

The `stream_id` is extracted separately. All three failures must report the
same boundary. A single error code is insufficient.

## Fixed decisions

- **U60 succeeds 3/3 with all 60 effective sizes:** GL-S0 was an old-stack
  limitation. Stop GraphBudget and do not run GB-G0O.
- **U60 fails 3/3 with the composite signature and one common boundary, while
  all three same-device defaults recover:** prospectively confirm the
  cross-runtime resource cliff. Only then may GB-G0O be designed.
- **Success/failure is mixed, the boundary changes, or recovery depends on
  order:** stop for runtime instability; do not make a systems-paper claim.
- **TP1 succeeds while only the old TP4 path failed:** the observation does not
  support a general GraphBudget abstraction; stop.
- **Stack mismatch, default startup failure, missing output, or an unregistered
  U60 failure:** technically invalid, with no positive interpretation.

Even a positive GB-Q0 result does not create the GraphBudget repository or
authorize a router. It only unlocks a separately frozen replicated-versus-
sharded fleet oracle.
