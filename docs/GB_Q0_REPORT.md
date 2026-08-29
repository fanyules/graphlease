# GB-Q0 official-stack qualification report

Status: **STOP GRAPHBUDGET — GL-S0 was an old-stack / old-TP-path limitation**

Decision date: 2026-08-29

Protocol commit: `e8a7b8c0dc5ff0ff703b73954c99655c273aa948`

Freeze commit: `9493441f8d2f11d38730e55bdd0065ff021aaef2`

## Decision

The current official stack captured U60 completely in all three fresh Qwen3-14B
BF16 TP1 processes. The frozen adjudicator returned:

```text
stop_graphbudget_old_stack_limitation
```

GraphBudget stops. GB-G0O, a router, complementary portfolio placement, and a
new GraphBudget repository are not created. The prior GL-S0 frozen decision is
unchanged; its `207005 / stream_id=1984` evidence remains a valid discovery of
the old stack's resource boundary, not evidence for a current general-purpose
fleet abstraction.

## Qualified stack

| Component | GB-Q0 value |
| --- | --- |
| Ascend driver / HDK | 25.5.1 |
| Firmware | 7.8.0.6.201, package 25.5.1 |
| CANN / NNAL | 9.1.0 |
| PyTorch / torch-npu | 2.10.0 / 2.10.0.post4 |
| vLLM / vLLM-Ascend | 0.23.0 / 0.23.0 |
| Official arm64 image | `quay.io/ascend/vllm-ascend:v0.23.0` |
| Image config ID | `sha256:660ce23a83574fdcd28ba126b07615829c06770c06fbdf6533f03cc4d90acdb1` |

This is the stable compatibility row published by vLLM-Ascend. Its release
notes state that HDK 25.5.1+ removes the old FULL_AND_PIECEWISE stream-budget
limitation:

- <https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html>
- <https://docs.vllm.ai/projects/ascend/en/main/user_guide/release_notes.html>

The official image reports optional profiler/GUI dependency conflicts under
`pip check`. They were retained and not repaired. The core torch-npu, vLLM,
vLLM-Ascend, ACL graph observer, eight-device visibility, and model asset all
passed direct import/runtime checks.

## Frozen matrix

| Configuration | Devices | Success | Effective sizes | Wrappers | Entries | KV tokens | NPU graph memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| runtime default | 0, 1, 2 | 3/3 | 19 | 42 | 798 | 135,424 | 0.62 GiB |
| U60 | 0, 1, 2 | 3/3 | 60 | 42 | 2,516 | 135,424 | 0.78 GiB |

Every process validated the complete stack before engine initialization and
issued zero requests. U60 completed both 60/60 PIECEWISE captures and the 56
eligible FULL decode captures. No capture-size filtering, `207005`, SQ/CQ
allocation failure, OOM, or ownership failure occurred.

Because all U60 processes succeeded, no default recovery process was required
or run. This is the preregistered order, not missing evidence.

Process-to-ready time was diagnostic only. The medians were 81.117 seconds for
default and 81.961 seconds for U60; the first default was a cold compilation
outlier. These timings are not performance evidence and do not enter the Gate.

## Environment transition record

The host began at driver/firmware package 25.2.3. Before upgrading, the driver,
firmware, DCMI, kernel modules, configuration, and installation logs were
archived. Driver 25.5.1 required one reboot before devices were available for
the firmware installer. Firmware 7.8.0.6.201 then upgraded all eight chips and
was activated by a second reboot. Final `npu-smi` reported eight healthy and
idle 910B4-1 devices.

The failed pre-reboot firmware attempt is retained in the raw archive; it is an
installation-order diagnostic, not a GB-Q0 run.

## Interpretation boundary

GB-Q0 supports this conclusion:

> The GL-S0 ACL resource cliff does not survive the current official stack in
> the preregistered Qwen3-14B TP1 qualification; a general GraphBudget system is
> therefore not justified.

It does not establish that every large graph set is free, nor does it provide
serving performance evidence. The narrower old-stack issue may be suitable for
an upstream compatibility note, not a systems paper.

No TP4 rerun, workload replay, GB-G0O oracle, profiling/pruning comparison,
router, or performance experiment was executed.

## Evidence

- Protocol: `docs/GB_Q0_PROTOCOL.md`
- Frozen configuration: `configs/gb_q0.json`
- Freeze: `results/gb_q0/GB_Q0_FREEZE.json`
- Decision: `results/gb_q0/GB_Q0_DECISION.json`
- Manifest: `results/gb_q0/GB_Q0_MANIFEST.sha256`
- Raw JSON, logs, and upgrade trace: `results/gb_q0/GB_Q0_RAW.tar.gz`
- Raw archive SHA-256:
  `85751f97c6117a0fe7a05326fda91fc958c886cd83d7a93a762ce0d434124537`
