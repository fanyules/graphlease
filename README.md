# GraphLease

GraphLease tests whether an LLM-serving replica should change its resident graph
portfolio as workload phases change, and whether the resource and restart costs
are large enough to justify an explicit cross-runtime graph lifecycle.

Gate 0 uses only Qwen3-1.7B BF16 TP1 replicas on A100 and Ascend 910B. It does
not implement a controller, modify vLLM/vLLM-Ascend, write a kernel, or use the
1 GbE link as a token hot path.

- Protocol: `docs/G0_PROTOCOL.md`
- Frozen configuration: `configs/g0.json`
- Results: `results/g0/`

G0-P passed exact within-platform plan invariance on A100 and 910B. G0-M then
failed to find a stable phase-dependent optimum, meaningful portfolio scarcity,
or a static-to-oracle gap. GraphLease is stopped and G0-L was not run. See
`docs/G0P_REPORT.md` and `docs/G0M_REPORT.md`.

An independent resource-only scale-transfer probe, GL-S0, keeps that failure
unchanged while testing whether the original 60-size union becomes physically
scarce for Qwen3-14B TP4. It issues no requests and cannot restore the
GraphLease controller claim. See `docs/GL_S0_PROTOCOL.md` and
`configs/gl_s0.json`.

GL-S0 completed with a cross-runtime resource asymmetry: A100 captured the
60-size union in 3/3 processes, while 910B failed 3/3 times at the same ACL
SQ/CQ allocation boundary (`stream_id=1984`). The frozen automated classifier
did not register error code `207005`, so its technical-invalid verdict is
retained alongside a separate semantic audit. See `docs/GL_S0_REPORT.md`.

GB-Q0 is the only active follow-up. It replaces the old stack with the current
official vLLM-Ascend compatibility row and runs Qwen3-14B BF16 TP1 as a
zero-request stack qualification. It does not run performance or create a
GraphBudget system. See `docs/GB_Q0_PROTOCOL.md`.

Local checks:

```bash
python -m unittest discover -s tests -v
python scripts/freeze_g0.py --output /tmp/graphlease-freeze.json
python scripts/freeze_gl_s0.py --output /tmp/gl-s0-freeze.json
python scripts/freeze_gb_q0.py --output /tmp/gb-q0-freeze.json
```
