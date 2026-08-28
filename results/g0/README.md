# GraphLease Gate-0 results

No accelerator output is valid before the public freeze. Failed initialization,
OOM, stream exhaustion, correctness mismatch, and incomplete runs are retained.

G0-P must pass on both platforms before G0-M. G0-L is conditional on G0-M
answering the first three research questions positively.

G0-P passed. Its eight valid rows are in `preflight/`, construction failures are
in `attempts/`, and the machine verdict is `G0P_DECISION.json`.

G0-M completed all 24 formal runs and both union probes, then returned
`stop_no_stable_phase_dependent_optimum`. `G0M_DECISION.json` is the machine
adjudication. `G0M_RAW.tar.gz` is the complete verified formal/resource/log
archive and `G0M_RAW.sha256` records its digest. G0-L was not run.
