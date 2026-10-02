# Eval results (2026-09-25_1341)

Pipeline model: claude-sonnet-5. Judge: claude-opus-5. One live run per company; web research varies between runs.

**Coverage:** this run covers 2 of the planned 6 companies due to budget. Skipped: Scout AI, Stripe, Databricks, Notion.

**Sony Play Station:** pipeline state reused from `evals/results/2026-09-25_1218/states/sony_play_station.json`, not re-run. Its pipeline cost was spent in that earlier run and isn't in the Cost column.

**Providers `unrecorded`:** the trace predates provider recording.

| Company | Claims (verified / unverified / contradicted / from posting / off-target) | Verified precision | Budget misses | Truly unconfirmable | Contradicted agreement | From-posting precision | Report faithfulness | Report consistency (1 = no contradictions) | Unverified stated as fact | Planner F1 | Providers (NIM claims excluded) | Time | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sony Play Station | 2 / 27 / 0 / 4 / 3 | 2/2 | 1/5 | 4/6 | n/a | 3/4 | 0.97 | 0.97 | 1 | 1.00 | unrecorded (0) | reused | $1.09 (judge only) |
| CaseGuard | 5 / 24 / 1 / 0 / 0 | 5/5 | 5/5 | 1/2 | 0/1 | n/a | 1.00 | 1.00 | 2 | 1.00 | claude (0) | 253s | $1.45 |
