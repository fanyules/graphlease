# GL-S0 Qwen3-14B TP4 resource audit report

Status: **cross-runtime resource mechanism observed; frozen automated classifier invalid**

Decision date: 2026-08-29  
Pre-output code commit: `459e94077c7fe2fefe271513edaeb94827a97336`  
Freeze commit: `a19e574c019b16634fb68bb35c6c9de98bb0f52c`

## Decision

GL-S0 observed the resource asymmetry that motivated the scale-transfer probe:
the A100 runtime captured the complete 60-size union in all three fresh
processes, while the 910B runtime failed in all three fresh processes at the
same ACL stream-allocation boundary. This is evidence for a separate
cross-runtime GraphBudget hypothesis. It does not restore the failed
GraphLease controller claim.

The frozen automated adjudicator reports
`technical_invalid_or_unregistered_union_failure`. That result is retained.
The classifier whitelist registered error code `207008` and two public error
strings, whereas the runtime returned `207005` with the more specific messages
`Alloc sq cq fail`, `resource alloc fail`, and `stream_id=1984`. Because the
classifier was frozen before output, it is not patched after the fact. The
physical resource event and the classifier-contract defect are reported as
separate facts.

## Frozen matrix result

| Platform / portfolio | Fresh processes | Effective sizes | Captured entries per rank | KV capacity | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| A100 default | 3/3 | 35 | 1,454 | 632,160; 637,088; 637,088 | success |
| A100 60-size union | 3/3 | 60 | 2,516 | 632,160; 637,088; 637,088 | success |
| 910B default | 3/3 | 19 | 798, with 42 wrappers | 1,080,192 in all runs | success |
| 910B 60-size union | 0/3 | capture aborted | no post-init snapshot | unavailable | SQ/CQ allocation failure at stream 1,984 |

The held-out scaling prediction was unusually precise. The 14B 910B default
run exposed 42 wrappers rather than the 1.7B model's 30, and materialized
`19 x 42 = 798` entries. On A100, the union completed with 2,516 captured
entries, equal to the preregistered estimate `60 x 42 - 4`. On 910B, all three
union runs reached the same `AclmdlRICaptureBegin` failure before engine-ready:

- error code `207005`;
- `Failed to apply for memory`;
- `[SqCqManage] Alloc sq cq fail`;
- `stream_id=1984`, `retCode=0x7020023`.

The three failing processes lasted 96.00, 84.92, and 85.75 seconds. The three
default processes completed in 114.61, 105.23, and 103.77 seconds and reported
about 0.52 GiB of NPU graph memory per rank. This excludes a default TP4
compatibility failure and isolates the failure to additional capture breadth.

## A100 resource result

A100's union increased the captured-entry count from 1,454 to 2,516 without a
paired KV-capacity loss and without an observable increase in the allocator's
post-init reserved bytes. Median process-to-ready time increased by 21.70%
(paired increases 8.68%, 27.24%, and 21.70%). Startup-only cost is not a
positive scarcity result under the protocol, but A100's complete capture is
the required control for the 910B failure.

## Interpretation boundary

The old result remains:

```text
G0-M = failed_no_stable_phase_optimum_no_scarcity_no_oracle_gap
GraphLease controller = not restored
```

GL-S0 supports only this new statement:

> At Qwen3-14B TP4, an identical logical 60-size graph portfolio is feasible on
> the tested CUDA runtime but repeatedly exhausts ACL SQ/CQ stream resources on
> the tested CANN runtime.

It does not show that a natural workload changes its best portfolio, that a
strong static configuration leaves an oracle gap, that a black-box router is
insufficient, or that lifecycle switching amortizes drain/capture/rejoin. A
new GraphBudget Gate must establish those conditions prospectively. The known
Qwen3-14B TP4 output instability on 910B remains an independent blocker for
performance evidence; it was irrelevant to this zero-request resource audit.

No extra restart, capture size, workload, model, kernel, router, or controller
was run after inspecting GL-S0 output.

## Evidence

- Frozen protocol and configuration: `docs/GL_S0_PROTOCOL.md`,
  `configs/gl_s0.json`
- Frozen mechanical decision: `results/gl_s0/GL_S0_DECISION_FROZEN.json`
- Explicit semantic audit: `results/gl_s0/GL_S0_SEMANTIC_AUDIT.json`
- Per-file manifest: `results/gl_s0/GL_S0_MANIFEST.sha256`
- Compressed raw JSON and logs: `results/gl_s0/GL_S0_RAW.tar.gz`
- Raw archive SHA-256:
  `7a57242c6c0ff8d94159f74a6a86fa932ed5d648d208c70c273b023f185db498`
