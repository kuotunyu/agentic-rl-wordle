---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- reinforcement-learning
- grpo
- agentic-rl
- multi-turn
- wordle
- peft
- lora
language:
- en
---

# Qwen2.5-1.5B Wordle GRPO LoRA adapter

## Repository role

This repository contains a **LoRA adapter**, not standalone full-model weights. It was trained
from the base repository identity `Qwen/Qwen2.5-1.5B-Instruct` for a multi-turn Wordle environment
using GRPO. The adapter must be loaded together with the compatible base model.

The PEFT configuration uses rank 16, alpha 32, dropout 0.05, and causal language modeling as the
task type. The current `adapter_model.safetensors` LFS SHA-256 is
`92e6379ed7ddf363e7f500b143afa7a2dc725d3e86bd87bc9eb933831c7d68b7`.

## Evaluation

The reported tuned evaluation is the adapter evaluation. It is a paired, greedy evaluation over
the complete held-out 463-word split with seed 42. Only committed aggregate evidence is available.

| Measure | Base | Tuned LoRA adapter |
|---|---:|---:|
| Wins | 0/463 | 13/463 = 2.81% |
| Wilson 95% CI | 0.00%–0.82% | 1.65%–4.74% |
| Protocol adherence | 0% | 2749/2753 = 99.85% |
| Legal actions | 0% | 2748/2753 = 99.82% |

The paired discordance was 0 base-only wins and 13 tuned-only wins. The two-sided exact paired
McNemar result is `0.000244140625`; treating the interim and final evaluations as two looks gives
the conservative Bonferroni-adjusted value `0.00048828125`.

Strategy use remained weak: absent letters were reused on 1340/2290 turns with information, and
known green positions were broken on 1119/2290 such turns.

**Protocol learning succeeded; strategy learning remained limited; this is not a practical Wordle solver.** The 2.81% win rate is statistically distinguishable from this base evaluation,
but it is not evidence of practical solving capability or general RL superiority.

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
- The aggregate result does not prove full strategy learning, production readiness, or complete
  exclusion of reward exploitation.

## Loading

Load the compatible base model first, then attach this LoRA adapter with PEFT. Because the exact
historical upstream base revision is unavailable, do not present a contemporary Qwen revision as
the cryptographically proven training revision.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "steven0226/qwen2.5-1.5b-wordle-grpo")
```
