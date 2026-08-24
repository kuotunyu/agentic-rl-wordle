---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
pipeline_tag: text-generation
tags:
- reinforcement-learning
- grpo
- agentic-rl
- multi-turn
- wordle
- merged-model
language:
- en
---

# Qwen2.5-1.5B Wordle GRPO merged full model

## Repository role

This repository contains the historical **merged full model** counterpart of the Wordle GRPO
LoRA adapter. It does not require a separately attached LoRA adapter for loading. The base
repository identity is `Qwen/Qwen2.5-1.5B-Instruct`; the current `model.safetensors` LFS SHA-256 is
`b6c55086e798e1f62e6d970f07ee97ab39c1e0af3ee4b6ecdb2a349e485087af`.

The repository relationship is supported by historical documentation, subject to the lineage
limits below. It is not a cryptographically complete adapter-to-merged derivation record.

## Evaluation

The published 463-word result originates from the adapter evaluation. No independent 463-word evaluation was run against these merged bytes, so the adapter result is not an independent replication of merged-model performance. The aggregate adapter result is included here only to
document the paired project evidence without inventing a separate merged-model result.

| Measure | Base | Tuned LoRA adapter evidence |
|---|---:|---:|
| Wins | 0/463 | 13/463 = 2.81% |
| Wilson 95% CI | 0.00%–0.82% | 1.65%–4.74% |
| Protocol adherence | 0% | 2749/2753 = 99.85% |
| Legal actions | 0% | 2748/2753 = 99.82% |

The paired discordance was 0 base-only wins and 13 tuned-only wins. The two-sided exact paired
McNemar result is `0.000244140625`; treating the interim and final evaluations as two looks gives
the conservative Bonferroni-adjusted value `0.00048828125`.

The adapter evaluation also recorded 1340/2290 absent-letter reuses and 1119/2290 broken green
positions on turns with information.

**Protocol learning succeeded; strategy learning remained limited; this is not a practical Wordle solver.** The 2.81% adapter win rate must not be presented as an independently measured
merged-model win rate, practical capability, or general RL superiority.

## Evidence links

- Immutable research/evidence source:
  https://github.com/kuotunyu/agentic-rl-wordle/commit/1a077a45e309594e5bb43743a8b84d89155595d4
- Stable source release URL:
  https://github.com/kuotunyu/agentic-rl-wordle/releases/tag/v1.0.0
- Aggregate report: `results/full_463_report.json`
- Recomputed analysis: `results/full_463_analysis.json`
- Public claim mapping: `docs/claim-matrix.md`

## Limitations

- Only aggregate 463-game evidence is committed; full per-episode records are unavailable.
- The historical GPU environment is not bit-for-bit reconstructable.
- The exact upstream Qwen commit was not preserved.
- The adapter-to-merged command and manifest were not preserved, and the historical record lacks
  an end-to-end run→code→prompt→bundle→model cryptographic chain. The relationship is documentary lineage, not complete cryptographic proof.
- The cfreshman word lists are fetch-only and have no explicit license. Apache-2.0 does not license the cfreshman word lists.
- Representative transcripts are illustrations, not a complete raw evaluation corpus.
- No independent evaluation of these merged bytes is available; do not infer byte-level
  equivalence or independently replicated performance from the documentary relationship.

## Loading

This repository is intended to load as a full causal language model and does not need a separate
PEFT adapter attachment. Because the exact historical upstream base revision and merge manifest
are unavailable, do not claim a complete cryptographic reconstruction of the merge.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "steven0226/qwen2.5-1.5b-wordle-grpo-merged"
)
```
