# GL-S0 results

This directory is reserved for the independently frozen Qwen3-14B TP4
resource-only scale-transfer probe. The old `results/g0/` evidence and G0-M
decision remain unchanged.

The matrix is complete. Raw per-process JSON and logs are retained in
`GL_S0_RAW.tar.gz`; `GL_S0_MANIFEST.sha256` authenticates the 12 JSON files,
12 logs, and frozen decision. The frozen automated decision and the explicit
semantic audit are separate artifacts because the registered error whitelist
omitted the observed ACL resource-allocation code `207005`.
