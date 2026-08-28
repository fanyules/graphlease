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

G0-P passed exact within-platform plan invariance on A100 and 910B. The formal
G0-M matrix is unblocked; see `docs/G0P_REPORT.md`.

Local checks:

```bash
python -m unittest discover -s tests -v
python scripts/freeze_g0.py --output /tmp/graphlease-freeze.json
```
