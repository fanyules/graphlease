# G0-M evidence import note

The complete node1 evidence archive was created after both platform queues
finished and was verified as:

```text
42d48c0f0af06ff3dc4c6f4203969efeddd27e661b500ee55e3a8b98178e7752  G0M_RAW.tar.gz
```

The first Windows extraction attempt exhausted C: while overwriting the already
downloaded machine decision. That incomplete local overwrite was briefly
committed. The canonical decision was immediately restored byte-for-byte from
the verified node1/E: copy; its SHA-256 is:

```text
b7444191ef1f89e1ddff515e28d1aacf48e68df04c0113f3b1f0dce2e990bdb9  G0M_DECISION.json
```

No remote run, input, result, metric, or verdict changed. To avoid storing about
130 MB of expanded repetitive token IDs in the Windows working tree, the GitHub
artifact retains all 24 formal JSON files, 26 logs, two resource JSON files, and
the machine decision in the verified compressed archive.
