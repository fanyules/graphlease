# G0 runner repair before valid preflight evidence

The first eager attempts on both platforms produced failed engineering artifacts
and no Gate-admissible row.

- A100 failed during vLLM warm-up because the selected Python environment does
  not contain a `ninja` executable. The environment handoff already specifies
  `/root/miniconda3/envs/qwen36-shard/bin` as the PATH prefix for vLLM 0.23
  FlashInfer JIT. The repaired runner now fails early if `ninja` is absent, and
  all A100 commands use that existing prefix.
- 910B initialized successfully, then the timing driver failed because public
  `LLM.enqueue()` deliberately changes sampling output to `FINAL_ONLY`. That API
  cannot expose first-token timing. The repaired driver uses public
  `LLMEngine.add_request()` with `RequestOutputKind.CUMULATIVE`, which preserves
  the same prompts, greedy sampling, seeds, scheduler, and model execution while
  making TTFT observable.

The same public raw-prompt path returns an internal suffixed ID from
`add_request()` while streamed `RequestOutput.request_id` retains the caller ID.
The driver therefore keys request timing by the caller ID. Both forms were
observed directly in the retained failed artifacts; no scheduler behavior was
changed.

No workload trace, capture size, threshold, model, runtime, or device changed.
The failed attempts remain under `results/g0/attempts/` and cannot enter any Gate
metric. A new canonical freeze must precede the repaired preflight.
