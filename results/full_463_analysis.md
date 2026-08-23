# Full 463-word paired analysis

- Fixed held-out split: `eval_full`, seed `42`, 463 paired answers
- Wins: base 0/463 → GRPO 13/463
- Observed absolute win-rate gain: **2.8%**
- Wilson 95% win-rate intervals: base [0.0%, 0.8%]; GRPO [1.6%, 4.7%]
- Exact paired McNemar: `p=0.000244`
- Conservative correction for 2 nested looks (n=200, then n=463): Bonferroni `p=0.000488`
- Recovered turn counts: protocol-adherent 2749/2753; legal 2748/2753; repeats 1/2753
- Recovered information-turn counts: absent-letter reuse 1340/2290; green-position breaks 1119/2290

## Capability funnel

| Capability | Base | GRPO LoRA |
|---|---:|---:|
| Tag adherence | 0.0% | **99.9%** |
| Legal action rate | 0.0% | **99.8%** |
| Preserve excluded letters | not defined | 41.5% |
| Preserve known green positions | not defined | 51.1% |
| Win within 6 turns | 0.0% | **2.8%** |

## Interpretation

The full held-out evaluation clears the project's statistical success criterion, even after a conservative two-look correction. The practical task-success rate remains small: GRPO reliably learned the interaction protocol, but only partially learned multi-turn constraint tracking and Wordle strategy.

The 463-word evaluation contains the earlier 200-word subset; it is the final, larger evaluation rather than an independent replication.

Evidence boundary: these values are recomputed from the committed aggregate JSON. Full per-episode records are not committed, so the per-turn source rows cannot be independently re-aggregated.
