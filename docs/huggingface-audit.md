# Hugging Face evidence audit

Audit basis date: 2026-08-24.

State: **`HF_README_ONLY_UPDATE_VERIFIED`**.

Phase C revalidated the authenticated account, exact parent revisions, README bytes, and complete
inventories before mutation. It then created one optimistic `README.md`-only commit in each model
repository, adapter first, and verified both complete post states before Phase D recorded them.

## Immutable Git linkage

- Public repository: `https://github.com/kuotunyu/agentic-rl-wordle`
- Immutable research/evidence source:
  `1a077a45e309594e5bb43743a8b84d89155595d4`
- Evidence source URL:
  `https://github.com/kuotunyu/agentic-rl-wordle/commit/1a077a45e309594e5bb43743a8b84d89155595d4`
- Future stable release URL:
  `https://github.com/kuotunyu/agentic-rl-wordle/releases/tag/v1.0.0`

The evidence source is public and immutable, but it is not the final release commit. The final
Git release evidence below records the verified HF post-update revisions without changing the
research/evidence source.

## Verified Phase C closure

Post-update adapter revision: `e95fc44d5914d800483a847e8768b86f33719f12`

Post-update merged revision: `94c0524cf963f7b22f1dc253eda5b4ef5a075956`

Post-update adapter README SHA-256: `ab9c473a8eb9efaf2ddc32873d405bb5eb6b5e305a0dddcdf774f8b7a77a0e6b`

Post-update merged README SHA-256: `b9f67c27188839000385ec85900a1d6825157aec1516af883349b7e52efb8e47`

Phase C public-safe receipt SHA-256:
`c4e429d0c34ad32a515eaae541608107afaec1f9e4bc4afa16f24b9d9561bd2d`

The filename sets remained identical at eight files per repository. Every non-README blob ID,
ordinary size, LFS SHA-256, and LFS size remained identical; only the `README.md` blob ID and size
changed to the authoritative Git card bytes.

### Public-safe transaction method note

Initial temporary receipt serialization stopped because a broad substring check treated the legal
filename `tokenizer_config.json` as forbidden metadata. Both authorized HF README commits had
already completed and passed their full post-state verification. The final receipt was then built
from a read-only retrieval of those exact post states. There was no retry, rollback, second HF
commit, or non-README mutation.

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

## Verified pre/post inventories

These tables are the complete normalized receipt inventories. A repeated blob, size, and LFS
identity is byte-for-byte evidence that the corresponding non-README artifact did not change.

### Adapter inventory: 8 files before and after

| File | Pre blob | Post blob | Size, pre → post | LFS SHA-256 / size |
|---|---|---|---:|---|
| `.gitattributes` | `52373fe24473b1aa44333d318f578ae6bf04b49b` | `52373fe24473b1aa44333d318f578ae6bf04b49b` | 1,570 → 1,570 | — |
| `README.md` | `a2c7a02968a4566c344ebf92f8c45773fc7a8455` | `9d4ee63d905f56e36fc60b1893d3cbe6d31d4a50` | 7,246 → 3,632 | — |
| `adapter_config.json` | `b1de78b261b03a020391d839400ff5664a009fd9` | `b1de78b261b03a020391d839400ff5664a009fd9` | 1,105 → 1,105 | — |
| `adapter_model.safetensors` | `8e4ffe7ef1ec47f9361fb94ad53d5fad338129b0` | `8e4ffe7ef1ec47f9361fb94ad53d5fad338129b0` | 73,911,112 → 73,911,112 | `92e6379ed7ddf363e7f500b143afa7a2dc725d3e86bd87bc9eb933831c7d68b7` / 73,911,112 |
| `chat_template.jinja` | `bdf7919a96cfe43d50914a007b9c0877bd0ec27e` | `bdf7919a96cfe43d50914a007b9c0877bd0ec27e` | 2,507 → 2,507 | — |
| `tokenizer.json` | `34510ff0037cd50428af467a17ead5a96140a32c` | `34510ff0037cd50428af467a17ead5a96140a32c` | 11,421,892 → 11,421,892 | `3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8` / 11,421,892 |
| `tokenizer_config.json` | `4d8760d91bde2ac751d25844835c33847a68cdf9` | `4d8760d91bde2ac751d25844835c33847a68cdf9` | 749 → 749 | — |
| `training_args.bin` | `4ab7d01ba69f79c9a8b301793796f9272e000a87` | `4ab7d01ba69f79c9a8b301793796f9272e000a87` | 7,569 → 7,569 | `9df95d5c562cd327397015f3324e3627fd280725af4f10be014233835e778ab8` / 7,569 |

### Merged inventory: 8 files before and after

| File | Pre blob | Post blob | Size, pre → post | LFS SHA-256 / size |
|---|---|---|---:|---|
| `.gitattributes` | `52373fe24473b1aa44333d318f578ae6bf04b49b` | `52373fe24473b1aa44333d318f578ae6bf04b49b` | 1,570 → 1,570 | — |
| `README.md` | `581cb1e37f31b8d200c05576da0647eba12aa1ae` | `33db8ba6e1032f2c30f14e35dd4143ab9d654c2a` | 7,347 → 3,793 | — |
| `chat_template.jinja` | `bdf7919a96cfe43d50914a007b9c0877bd0ec27e` | `bdf7919a96cfe43d50914a007b9c0877bd0ec27e` | 2,507 → 2,507 | — |
| `config.json` | `97c2b63b467e3d0f1c22c493f19e81c2fd8b5318` | `97c2b63b467e3d0f1c22c493f19e81c2fd8b5318` | 1,373 → 1,373 | — |
| `generation_config.json` | `a8aca904d377977b666e4bd5d526356e627574bf` | `a8aca904d377977b666e4bd5d526356e627574bf` | 242 → 242 | — |
| `model.safetensors` | `d7d7779ec79579c35d69a7a0ca6ecdfec41c051a` | `d7d7779ec79579c35d69a7a0ca6ecdfec41c051a` | 3,087,467,144 → 3,087,467,144 | `b6c55086e798e1f62e6d970f07ee97ab39c1e0af3ee4b6ecdb2a349e485087af` / 3,087,467,144 |
| `tokenizer.json` | `34510ff0037cd50428af467a17ead5a96140a32c` | `34510ff0037cd50428af467a17ead5a96140a32c` | 11,421,892 → 11,421,892 | `3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8` / 11,421,892 |
| `tokenizer_config.json` | `770e41d6c92519d525eede4cbcf3ba27f6425311` | `770e41d6c92519d525eede4cbcf3ba27f6425311` | 694 → 694 | — |

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

The authorized transaction required account identity `steven0226`, exact expected parent
revisions, and full baseline matches. Each repository received exactly one `README.md` add
operation. Any filename, non-README blob, LFS SHA, or size change would have stopped release work.
If adapter had succeeded and merged failed, the required stop state was
`PARTIAL_HF_CARD_UPDATE`; no rollback, GitHub merge, tag, or Release would have been permitted.
