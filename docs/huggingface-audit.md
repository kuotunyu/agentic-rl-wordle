# Hugging Face evidence audit

Audit basis date: 2026-08-24.

State: **HF README-only update pending owner authorization**.

Phase A uses the owner-approved baseline below and performs no Hugging Face API operation. Phase C
must revalidate the authenticated account, live revision, and complete inventory for both
repositories before any mutation. No post-update revision is recorded here because no card update
has occurred.

## Immutable Git linkage

- Public repository: `https://github.com/kuotunyu/agentic-rl-wordle`
- Immutable research/evidence source:
  `1a077a45e309594e5bb43743a8b84d89155595d4`
- Evidence source URL:
  `https://github.com/kuotunyu/agentic-rl-wordle/commit/1a077a45e309594e5bb43743a8b84d89155595d4`
- Future stable release URL:
  `https://github.com/kuotunyu/agentic-rl-wordle/releases/tag/v1.0.0`

The evidence source is public and immutable, but it is not the final release commit. The final
Git release evidence can record HF post-update revisions only after the authorized README-only
transaction succeeds.

## Adapter baseline

- Repository: `steven0226/qwen2.5-1.5b-wordle-grpo`
- Role: LoRA adapter; requires a compatible `Qwen/Qwen2.5-1.5B-Instruct` base model
- Expected revision: `ef1e98ce214921049b86dce7c104c88875130023`
- Expected README content SHA-256:
  `c3ab2ecbc0a032e77345239b02f41b967a8398017e312d7a2ea8e45a04afcf5b`
- Expected README size: 7,246 bytes

| File | Expected blob identity | LFS SHA-256 / size when applicable |
|---|---|---|
| `.gitattributes` | `52373fe24473b1aa44333d318f578ae6bf04b49b` | — |
| `adapter_config.json` | `b1de78b261b03a020391d839400ff5664a009fd9` | — |
| `adapter_model.safetensors` | `8e4ffe7ef1ec47f9361fb94ad53d5fad338129b0` | `92e6379ed7ddf363e7f500b143afa7a2dc725d3e86bd87bc9eb933831c7d68b7` / 73,911,112 bytes |
| `chat_template.jinja` | `bdf7919a96cfe43d50914a007b9c0877bd0ec27e` | — |
| `README.md` | `a2c7a02968a4566c344ebf92f8c45773fc7a8455` | content SHA above / 7,246 bytes |
| `tokenizer_config.json` | `4d8760d91bde2ac751d25844835c33847a68cdf9` | — |
| `tokenizer.json` | `34510ff0037cd50428af467a17ead5a96140a32c` | `3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8` / 11,421,892 bytes |
| `training_args.bin` | `4ab7d01ba69f79c9a8b301793796f9272e000a87` | `9df95d5c562cd327397015f3324e3627fd280725af4f10be014233835e778ab8` / 7,569 bytes |

`adapter_model.safetensors`, `training_args.bin`, and every other non-README artifact are outside
the authorized mutation boundary.

## Merged baseline

- Repository: `steven0226/qwen2.5-1.5b-wordle-grpo-merged`
- Role: merged full model; no separately attached LoRA adapter is required
- Expected revision: `a59a4fb4c26e5d0612ce3a3574193ec58d46fc64`
- Expected README content SHA-256:
  `d3a35ef0db5324b3f67135e0cd216dbd980cd6afc0cf03aaf6621e36b9777e00`
- Expected README size: 7,347 bytes

| File | Expected blob identity | LFS SHA-256 / size when applicable |
|---|---|---|
| `.gitattributes` | `52373fe24473b1aa44333d318f578ae6bf04b49b` | — |
| `chat_template.jinja` | `bdf7919a96cfe43d50914a007b9c0877bd0ec27e` | — |
| `config.json` | `97c2b63b467e3d0f1c22c493f19e81c2fd8b5318` | — |
| `generation_config.json` | `a8aca904d377977b666e4bd5d526356e627574bf` | — |
| `model.safetensors` | `d7d7779ec79579c35d69a7a0ca6ecdfec41c051a` | `b6c55086e798e1f62e6d970f07ee97ab39c1e0af3ee4b6ecdb2a349e485087af` / 3,087,467,144 bytes |
| `README.md` | `581cb1e37f31b8d200c05576da0647eba12aa1ae` | content SHA above / 7,347 bytes |
| `tokenizer_config.json` | `770e41d6c92519d525eede4cbcf3ba27f6425311` | — |
| `tokenizer.json` | `34510ff0037cd50428af467a17ead5a96140a32c` | `3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8` / 11,421,892 bytes |

`model.safetensors` and every other non-README artifact are outside the authorized mutation
boundary.

## Card roles and evidence boundary

- `docs/model_card.md` is the authoritative LoRA adapter payload.
- `docs/model_card_merged.md` is the distinct authoritative merged full-model payload.
- The 13/463 result is the adapter evaluation. No independent 463-word evaluation was run against
  the merged bytes, so it is not an independent merged-model replication.
- Only aggregate 463-game evidence is committed; full per-episode records are unavailable.
- The historical GPU environment is not bit-for-bit reconstructable.
- The cfreshman word lists are fetch-only and have no explicit license; Apache-2.0 does not license
  those third-party lists.

The exact upstream Qwen commit was not preserved. The adapter-to-merged derivation is documentary lineage,
not complete cryptographic proof: the merge command, merge manifest, and an end-to-end
run→code→prompt→bundle→model chain were not preserved. A contemporary base revision must not be
substituted as historical evidence.

## Phase C fail-closed boundary

Phase C requires a new owner authorization, account identity `steven0226`, exact expected parent
revisions, and full baseline matches. Each repository may receive exactly one `README.md` add
operation. Any filename, non-README blob, LFS SHA, or size change stops release work. If adapter
succeeds and merged fails, the required stop state is `PARTIAL_HF_CARD_UPDATE`; no rollback,
GitHub merge, tag, or Release is permitted.
