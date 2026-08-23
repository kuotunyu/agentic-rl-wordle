# Hugging Face evidence audit

Read-only audit date: 2026-08-24. No model weights were downloaded and no Hugging Face state was
modified.

## Public revisions

| Repository | Public revision | Weight object metadata from HTTP HEAD |
|---|---|---|
| `steven0226/qwen2.5-1.5b-wordle-grpo` | `ef1e98ce214921049b86dce7c104c88875130023` | `adapter_model.safetensors`: 73,911,112 bytes; SHA-256/LFS ETag `92e6379ed7ddf363e7f500b143afa7a2dc725d3e86bd87bc9eb933831c7d68b7` |
| `steven0226/qwen2.5-1.5b-wordle-grpo-merged` | `a59a4fb4c26e5d0612ce3a3574193ec58d46fc64` | `model.safetensors`: 3,087,467,144 bytes; SHA-256/LFS ETag `b6c55086e798e1f62e6d970f07ee97ab39c1e0af3ee4b6ecdb2a349e485087af` |

Adapter metadata declares `Qwen/Qwen2.5-1.5B-Instruct`, PEFT 0.19.1, LoRA rank 16, alpha 32,
dropout 0.05, and causal-LM task type. Its base `revision` is null, so the exact upstream Qwen
commit is not preserved. The merged config identifies `Qwen2ForCausalLM` and transformers 5.12.1.

## Model-card and repository linkage findings

- Both public model cards are still the 2026-07-11 n=200 versions (`4/200`, not significant).
  Their README SHA-256 values are adapter
  `c3ab2ecbc0a032e77345239b02f41b967a8398017e312d7a2ea8e45a04afcf5b` and merged
  `d3a35ef0db5324b3f67135e0cd216dbd980cd6afc0cf03aaf6621e36b9777e00`.
- Local `docs/model_card.md` contains the later 13/463 aggregate and the required limited-strategy
  interpretation. It has not been pushed, per owner restriction.
- Both remote cards refer to a GitHub `agentic-rl-wordle` project, but this local Git repository
  has no remote and no public GitHub repository currently supplies a candidate SHA.
- The merged repository says it is the merged counterpart, but no public merge manifest records
  the exact adapter revision, exact upstream base revision, merge command, or output hash in one
  cryptographic chain. The two weight-object hashes above identify current public files but do not
  prove their derivation relationship.
- The ignored historical Colab bundle currently matches committed `wordle_rl_bundle.sha256`
  (`79979fff55f4d3abe91cd75dc509d7b7b50a664ecfc51bbc9e32bb9324c3a4bf`) and passes the
  publication-boundary scan. The July 2026 aggregate report itself does not embed a Git commit,
  prompt hash, bundle hash, or adapter revision, so end-to-end run→code→model linkage remains
  documentary rather than cryptographically self-contained.

## External publication actions

1. Create and publish `kuotunyu/agentic-rl-wordle`, then link its immutable v1.0.0 candidate SHA.
2. Sync the local full-463 model card to both HF repositories without changing weights.
3. Pin the exact upstream Qwen base revision and add a merge manifest linking base + adapter →
   merged weight hash.
4. Link the HF revisions above from the GitHub release evidence and link the GitHub candidate SHA
   back from both model cards.
5. For a future evaluation, include Git SHA, prompt hash, bundle hash, and model revisions directly
   in the raw/aggregate result metadata.
