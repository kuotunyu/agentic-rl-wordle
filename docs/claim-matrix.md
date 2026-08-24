# Claim matrix

This matrix maps portfolio-facing claims to immutable evidence and executable recomputation.
`results/full_463_report.json` is the committed aggregate produced by the GPU evaluation and is
not rewritten by release hardening. `results/full_463_analysis.*` is derived from it.

The immutable research/evidence source commit is
`1a077a45e309594e5bb43743a8b84d89155595d4`; it is not the final release commit. This matrix uses
the aggregate-only boundary and does not imply that full per-episode records are available.

| README/model-card claim | Source artifact | Recomputed value / test | Evidence boundary |
|---|---|---|---|
| Fixed held-out evaluation contains 463 answers at seed 42 | `full_463_report.json` `meta`; `docs/data-governance.md` | `test_words.py` proves 1,852/463, deterministic seed, complete disjoint split | Split is executable after the small pinned word fetch |
| Base 0/463; tuned 13/463 (2.81%) | `full_463_report.json` `rows.*.{n,wins}` | `test_committed_aggregate_recovers_action_counts_and_statistics` | Aggregate counts; full 463 per-episode records are not committed |
| Tuned win-rate Wilson 95% CI is 1.65%–4.74%; base interval is 0.00%–0.82% | `full_463_report.json` wins/n | `analyze_full_463.py` calls `wordle_rl.metrics.wilson_ci`; `test_analysis_recomputes_wilson_interval_instead_of_trusting_input` proves the stored CI is not trusted | Wilson interval is fully recomputable from counts |
| 99.85% protocol adherence | Aggregate `tag_ok_rate` | Recovered `2749/2753`; committed-artifact test | Integer denominator is recoverable from the exact serialized aggregate rate; raw turns are absent |
| 99.82% legal actions | Aggregate `illegal_rate` | Recovered `2748/2753`; committed-artifact test | Same aggregate-only boundary |
| Excluded-letter preservation is 41.5%; green-position preservation is 51.1% | Aggregate violation rates | Recovered violations `1340/2290` and `1119/2290`, hence preservation `950/2290` and `1171/2290` | Aggregate-only; representative failures appear in `full_463_report.md` |
| Repeat rate is 0.036% (displayed as 0.0%) | Aggregate `repeat_rate` | Recovered `1/2753` | “No observed loop pattern” is supported; “no reward hacking” would be too broad |
| Paired exact McNemar p=0.000244140625 | Base wins are zero; tuned wins are 13 on the same 463 answers | `exact_mcnemar_base_zero(13) = 2*(0.5^13)`; focused unit test | Recoverable because zero base wins makes every tuned win a one-direction discordance |
| Two nested looks Bonferroni p=0.00048828125 | Interim n=200 is a prefix of final n=463 | `min(1, 2*0.000244140625)`; focused unit test | Conservative multiplicity description, not an independent replication claim |
| Protocol learning succeeded; strategy learning remained limited | Counts above plus 58.5% absent-letter reuse, 48.9% green breaks, and 2.81% wins | `full_463_analysis.*`; representative transcripts in `full_463_report.md` | Supported conclusion; the 2.81% win rate is not a practical Wordle solver |
| Training reward -9.4→-3.2 and 3,000 steps | `docs/decision.md` historical run record | No committed raw `metrics.jsonl` recomputation | Reported historical evidence only; keep out of the primary externally verifiable headline unless redacted raw logs are published |

The strongest publication-safe headline is therefore: **GRPO produced near-complete protocol and
legal-action compliance, while held-out task success remained only 13/463 (2.81%) and multi-turn
constraint tracking remained weak.**
