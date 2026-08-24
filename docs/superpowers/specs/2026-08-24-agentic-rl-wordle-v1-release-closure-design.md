# Agentic RL Wordle v1.0.0 Release Closure Design

Status: **owner written review approved / implementation plan authorized**

Decision: **Approach A — staged, identity-preserving external closure**

Design date: **2026-08-24**

## 1. Purpose and terminal state

This design governs the final external-publication closure of `agentic-rl-wordle`. Its terminal
state is:

> `agentic-rl-wordle` → `v1.0.0` → **Frozen / Portfolio Complete**

The release delivers:

- Python distribution version `1.0.0`;
- a protected GitHub `main` branch;
- an annotated `v1.0.0` tag;
- a non-draft, non-prerelease, source-only GitHub Release with zero additional assets;
- distinct, updated Hugging Face model cards for the LoRA adapter and merged full model;
- unchanged Hugging Face weights and unchanged non-README Hugging Face artifacts;
- bidirectional GitHub/Hugging Face evidence links; and
- explicit preservation of every evaluation, provenance, licensing, and lineage limitation.

“Frozen / Portfolio Complete” describes the publication state of the portfolio repository. It is
not a production-readiness claim for the model.

The closure does not publish a PyPI distribution, Docker image, Hugging Face Space, dataset, new
model repository, or any additional GitHub Release asset. It does not execute training, GPU
evaluation, vLLM, or Colab workloads, and it does not download model weights.

## 2. Immutable research and claim contract

### 2.1 Commit roles

`1a077a45e309594e5bb43743a8b84d89155595d4` is the **immutable research/evidence source
commit**. It is not the final release commit.

Two descendants contain release-path hotfixes only:

- `300d75061985946a8585b24edbb86aabdeac943d` bounds Linux CUDA runtime discovery and removes
  unsafe recursive filesystem traversal.
- `8ce548b7b7ae2b812dbacadf477b6600e9d2d867` binds publication-identity scanning to the real
  branch tip in pull-request CI.

These hotfixes do not change research results, the 463-word aggregate, baseline artifacts,
training claims, reward/protocol behavior, or notebook research content. The release branch is
based on `8ce548b7b7ae2b812dbacadf477b6600e9d2d867`; cards and release evidence identify
`1a077a45e309594e5bb43743a8b84d89155595d4` as the evidence source.

### 2.2 Headline evidence

The following values are immutable release claims derived from committed aggregate evidence:

| Measure | Approved value |
|---|---:|
| Base wins | `0/463` |
| Tuned wins | `13/463 = 2.81%` |
| Base Wilson 95% CI | `0.00%–0.82%` |
| Tuned Wilson 95% CI | `1.65%–4.74%` |
| Protocol adherence | `2749/2753 = 99.85%` |
| Legal actions | `2748/2753 = 99.82%` |
| Exact paired McNemar p-value | `0.000244140625` |
| Two-look Bonferroni-adjusted p-value | `0.00048828125` |
| Excluded-letter reuse | `1340/2290` |
| Green-position breaks | `1119/2290` |

The only approved conclusion is:

> **Protocol learning succeeded; strategy learning remained limited; the 2.81% win rate is not a
> practical Wordle solver.**

The release must not claim a practical or strong Wordle solver, general RL superiority, complete
strategy learning, independent replication, full raw-record recomputation, production readiness,
or comprehensive exclusion of reward hacking. A mismatch between a headline and committed
aggregate evidence is resolved by lowering or removing the headline, never by changing evidence.

### 2.3 Evidence boundary

Only aggregate 463-game evaluation evidence is committed. Full per-episode records are
unavailable. The historical GPU environment is not bit-for-bit reconstructable. The aggregate
does not embed a complete Git→prompt→bundle→model identity chain. Representative transcripts are
illustrations, not a full raw corpus.

## 3. Publication identity graph

The closure uses the following acyclic identity graph:

1. The immutable evidence source commit exists publicly before any card update.
2. A release-source candidate is prepared on a branch descended from protected `main` and is
   exposed in an open pull request with green CI.
3. Both Hugging Face cards link the immutable evidence source commit and the future stable release
   URL; neither card needs an unknown final release commit SHA.
4. The card-only Hugging Face commits create two post-update revisions.
5. A later Git evidence-closure commit records those exact post-update revisions and all pre/post
   inventory receipts.
6. The final Git commit is integrated, then the annotated stable tag points to it, and the future
   release URL becomes live.

The stable release URL is
`https://github.com/kuotunyu/agentic-rl-wordle/releases/tag/v1.0.0`. The immutable evidence source
URL is
`https://github.com/kuotunyu/agentic-rl-wordle/commit/1a077a45e309594e5bb43743a8b84d89155595d4`.

This ordering prevents a cycle: Hugging Face depends only on an existing evidence commit and a
predeclared stable URL, while final Git evidence depends on already-created Hugging Face
revisions.

## 4. Git source contract

### 4.1 Planned release-source changes

The implementation is limited to these tracked paths:

| Path | Required release change |
|---|---|
| `README.md` | Replace stale publication status; use precise claims and public evidence links while retaining the practical-limit statement. |
| `docs/model_card.md` | Become the authoritative LoRA adapter card, including its exact remote payload. |
| `docs/model_card_merged.md` | New authoritative merged full-model card, distinct from the adapter card. |
| `docs/huggingface-audit.md` | Record pre-update baselines, post-update revisions, complete pre/post inventories, README hashes, unchanged identities, and unresolved lineage. |
| `docs/claim-matrix.md` | Align display precision and link every public claim to the committed aggregate and recomputation tests. |
| `docs/release-readiness.md` | Replace obsolete external blockers with completed or explicitly unresolved gates and receipts. |
| `src/wordle_rl/__init__.py` | Change the single version source from `1.0.0rc1` to `1.0.0`. |
| `.github/workflows/ci.yml` | Update version assertions/artifact name and pin every third-party action to an approved full commit SHA. |
| `CHANGELOG.md` | Record the stable source release, both hotfixes, card-only publication, and unchanged research evidence. |
| `release/v1.0.0.md` | Become final release notes containing Git/HF identities and all evidence/lineage/license limits. |
| `tests/test_release_closure.py` | Add deterministic release-contract tests for version consistency, action pins, separate card roles, required claims/limits, and stable URLs. |
| `docs/superpowers/specs/2026-08-24-agentic-rl-wordle-v1-release-closure-design.md` | Preserve this approved design. |
| `docs/superpowers/plans/2026-08-24-agentic-rl-wordle-v1-release-closure.md` | Define the exact execution checklist after written owner review of this spec. |

`pyproject.toml` remains unchanged under this design. Its version is already dynamic from
`wordle_rl.__version__`, and its author, license, Python range, and project/model URLs are already
correct. The built metadata must nevertheless be verified as `1.0.0`. Any newly discovered need
to alter `pyproject.toml` requires a written design amendment before implementation.

No other production code or test file is changed. In particular, training, evaluation, rewards,
protocols, datasets, word lists, notebooks, and historical commits remain untouched.

### 4.2 Immutable Git artifacts

The following classes are byte-invariant across the release branch:

- `results/full_463_report.json` and its Markdown counterpart;
- `results/full_463_analysis.json` and its Markdown counterpart;
- all baseline artifacts and historical reports;
- training/evaluation notebooks and the historical bundle identity;
- reward, protocol, rollout, training, and evaluation implementation; and
- word-list data, source pins, split definition, and seed.

If release closure requires altering any of these artifacts, the release stops.

### 4.3 Version and workflow identity

The distribution version is exactly `1.0.0` in the source package, built wheel, built sdist,
isolated imports, CLI environments, CI assertions, artifact name, changelog, readiness record, and
release notes.

Third-party actions are pinned to these verified commits:

- `actions/checkout@11d5960a326750d5838078e36cf38b85af677262`
- `actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065`
- `actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02`

The six required checks remain:

- `quality`
- `build-artifacts`
- `test-install (3.11, editable)`
- `test-install (3.11, regular)`
- `test-install (3.12, editable)`
- `test-install (3.12, regular)`

## 5. Distinct authoritative model cards

Both authoritative card files use UTF-8 without a byte-order mark and LF line endings. The remote
README payload is the canonical Git blob bytes from the pull-request head, not a platform-specific
working-tree conversion. After card publication, the corresponding local card and remote
`README.md` must be byte-identical.

### 5.1 Adapter card contract

`docs/model_card.md` is the authoritative card for
`steven0226/qwen2.5-1.5b-wordle-grpo`. It must state:

- the repository contains a LoRA adapter, not standalone full-model weights;
- the base repository identity is `Qwen/Qwen2.5-1.5B-Instruct`;
- the adapter must be loaded together with the compatible base model;
- LoRA rank is 16, alpha is 32, dropout is 0.05, and the task type is causal language modeling;
- the reported tuned evaluation is the adapter evaluation;
- the immutable evidence source commit and future `v1.0.0` release URL;
- the complete aggregate-only, environment, word-list-license, and lineage limitations; and
- the approved metrics and constrained conclusion in Section 2.

Its Hugging Face metadata retains `license: apache-2.0`, identifies the base repository, and uses
adapter/LoRA tags. Apache-2.0 applies to the released project/model materials that carry it; the
card explicitly states that it does not license the fetched cfreshman word lists.

### 5.2 Merged card contract

`docs/model_card_merged.md` is the authoritative card for
`steven0226/qwen2.5-1.5b-wordle-grpo-merged`. It must state:

- the repository contains merged full-model weights and does not require a separately attached
  LoRA adapter for loading;
- the published 463-word result originates from the adapter evaluation;
- no independent 463-word evaluation was run against the merged bytes;
- the adapter result is not an independent replication of merged-model performance;
- the repository is the historical merged counterpart, subject to the documentary-lineage limit;
- the immutable evidence source commit and future `v1.0.0` release URL;
- the complete aggregate-only, environment, word-list-license, and lineage limitations; and
- the approved metrics and constrained conclusion in Section 2.

Its Hugging Face metadata retains `license: apache-2.0` and identifies a merged full model rather
than an adapter. It omits adapter-only loading claims and does not imply that Apache-2.0 covers the
cfreshman word lists.

### 5.3 Shared lineage wording

Both cards may state that the base repository identity is known and that current adapter and
merged weights are identifiable by SHA-256. They may state that the merged repository is the
counterpart produced by the historical workflow.

They must also state that the historical record does not preserve:

- the exact upstream Qwen commit;
- a complete original GPU environment lock;
- the adapter-to-merged merge command and manifest; or
- an end-to-end run→code→prompt→bundle→model cryptographic chain.

The derivation relationship is therefore documentary lineage, not complete cryptographic proof.
No contemporary Qwen revision may be substituted as the historical training revision, and no
retroactive merge manifest may be presented as historical evidence.

## 6. Hugging Face pre-update baseline

### 6.1 Adapter repository

- Repository: `steven0226/qwen2.5-1.5b-wordle-grpo`
- Expected revision: `ef1e98ce214921049b86dce7c104c88875130023`
- Expected README content SHA-256:
  `c3ab2ecbc0a032e77345239b02f41b967a8398017e312d7a2ea8e45a04afcf5b`
- Expected README blob: `a2c7a02968a4566c344ebf92f8c45773fc7a8455`

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

`adapter_model.safetensors` and `training_args.bin` are immutable. Every other non-README file is
also immutable.

### 6.2 Merged repository

- Repository: `steven0226/qwen2.5-1.5b-wordle-grpo-merged`
- Expected revision: `a59a4fb4c26e5d0612ce3a3574193ec58d46fc64`
- Expected README content SHA-256:
  `d3a35ef0db5324b3f67135e0cd216dbd980cd6afc0cf03aaf6621e36b9777e00`
- Expected README blob: `581cb1e37f31b8d200c05576da0647eba12aa1ae`

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

`model.safetensors` and every other non-README file are immutable.

## 7. README-only Hugging Face transaction

Before mutation, the authenticated Hugging Face identity must resolve to `steven0226` without
displaying credentials. Both repositories receive a fresh complete inventory containing revision,
filename, ordinary blob ID, LFS SHA-256, and size. The snapshot must exactly match Section 6.

The adapter update occurs first, followed by the merged update. Each operation creates one commit
whose only changed path is `README.md`; its payload is the corresponding authoritative Git card.
No repository clone or workflow may download, reserialize, upload, delete, or otherwise touch a
weight. Config, tokenizer, chat-template, training-argument, and attribute files are outside the
mutation boundary.

After each update, verification must prove:

- the repository revision changed only through the README commit;
- the filename set is identical to the pre-update inventory;
- the README content SHA-256 equals the authoritative Git card SHA-256;
- every non-README ordinary blob ID is unchanged;
- every LFS SHA-256 and size is unchanged; and
- the public card renders with the intended repository role and limitations.

The post-update revision, README blob ID, README content hash, full post-update inventory, and
verification result become public-safe receipts. Credentials, environment dumps, and local paths
do not become receipts.

## 8. Required four-phase sequence

### Phase 1 — Local release candidate

1. Use `codex/v1.0.0-release-closure` as the sole release branch, created from protected `main` at
   `8ce548b7b7ae2b812dbacadf477b6600e9d2d867`.
2. Prepare both distinct authoritative model-card payloads.
3. Apply only the Git source changes in Section 4, including version `1.0.0`, action pins, stable
   documentation, and the release-contract test.
4. Record the pre-update HF baseline without inventing post-update revisions.
5. Create a release-source candidate commit using the formal `kuotunyu` noreply identity.
6. Pass all local quality, test, build, install, analysis, and publication gates.
7. Publish the release branch and open a pull request against `main`.
8. Require the pull-request head to be a strict fast-forward descendant of `main` and all six CI
   checks to succeed.
9. Keep the pull request open; do not integrate it before HF closure.

### Phase 2 — Hugging Face README-only update

1. Revalidate authenticated account identity and both complete pre-update inventories.
2. Commit the adapter authoritative card as the adapter repository `README.md` only.
3. Verify its post-update inventory and immutable artifacts.
4. Commit the merged authoritative card as the merged repository `README.md` only.
5. Verify its post-update inventory and immutable artifacts.
6. Capture both post-update revisions and public-safe receipts.

If the adapter update succeeds and the merged update fails, the terminal state is
`PARTIAL_HF_CARD_UPDATE`. All receipts are preserved; GitHub merge, tag, and Release are forbidden;
there is no force-push, automatic rollback, or mutation of the successful adapter card. Execution
stops for owner direction.

### Phase 3 — Immutable HF closure commit

1. Write both post-update revisions, pre/post inventories, README identities, and unchanged
   non-README identities into the Git HF audit and release evidence.
2. Verify each authoritative Git card is byte-identical to its remote README.
3. Create one docs/evidence closure commit on the same release branch.
4. Publish the updated branch and require all six pull-request CI checks to succeed again.
5. Freeze the release-source tree and both HF cards. No further source or HF card change is
   permitted after this commit without restarting closure from an owner-approved design revision.

### Phase 4 — GitHub stable closure

1. Advance `main` to the exact release pull-request head by fast-forward only; no merge commit,
   squash, rebase, cherry-pick, force-push, or history rewrite is allowed.
2. Require the exact final `main` SHA to complete all six push CI checks successfully.
3. Revalidate branch protection: strict required checks, administrator enforcement, linear history,
   force-push disabled, deletion disabled, and no fictitious reviewer requirement.
4. Create an annotated `v1.0.0` tag that peels to the exact final `main` commit and uses the formal
   `kuotunyu` noreply identity.
5. Require tag CI on the exact peeled commit to complete all six checks successfully.
6. Create a non-draft, non-prerelease, source-only GitHub Release with zero uploaded assets.
7. Verify release notes contain final Git SHA, evidence source SHA, pre/post HF revisions, immutable
   weight identities, aggregate-only evidence, missing cryptographic lineage, and cfreshman license
   boundaries.
8. Only after every receipt is verified, remove the temporary release worktree and local/remote
   release branches. Preserve protected `main`, the annotated tag, and the GitHub Release.

## 9. Verification and acceptance gates

The release candidate and final release tree must pass:

- Ruff lint and format checks;
- aggregate recomputation against committed evidence;
- publication-boundary, credential, identity, privacy, artifact-size, and notebook-output scans;
- the complete CPU test suite, including release-contract tests;
- Python 3.11 and 3.12 editable and regular installs;
- wheel and sdist builds plus isolated import and both CLI smoke paths;
- exact distribution version and metadata comparison across source, wheel, and sdist;
- exact card-role, metric, limitation, URL, and byte-identity checks;
- proof that immutable research artifacts match the evidence source commit; and
- Git whitespace validation and a clean working tree.

A canceled, timed-out, skipped-required, neutral, or allowed-failure job is not success. Every
required job must complete with a successful conclusion on the exact PR head, final main SHA, and
tag peeled commit.

## 10. Stop-state model

| Condition | Required state and response |
|---|---|
| Git/GitHub baseline mismatch before work | `BASELINE_MISMATCH`; stop without repair or mutation. |
| HF identity, revision, README, inventory, blob, LFS hash, or size mismatch before mutation | `HF_BASELINE_MISMATCH`; do not update either card. |
| Adapter update fails before any remote commit | `HF_UNCHANGED`; preserve receipts and stop. |
| Adapter succeeds, merged fails | `PARTIAL_HF_CARD_UPDATE`; no rollback and no GitHub merge/tag/Release. |
| Any post-update non-README diff or identity change | `HF_ARTIFACT_INTEGRITY_FAILURE`; stop all release work and preserve evidence. |
| Remote card bytes differ from authoritative Git bytes | `HF_CARD_IDENTITY_FAILURE`; stop before GitHub integration. |
| Metrics fail aggregate recomputation or require research-artifact edits | `RESEARCH_EVIDENCE_BLOCKED`; stop without changing evidence or substituting an unreviewed claim. |
| Public-boundary, license, identity, build, test, or CI gate fails | `RELEASE_GATE_FAILED`; do not integrate or publish stable state. |
| PR head is not a fast-forward descendant of current main | `NON_FAST_FORWARD_CANDIDATE`; do not integrate. |
| Tag does not peel to exact final main or main/tag CI fails | `STABLE_CLOSURE_FAILED`; do not create or finalize the GitHub Release. |

No stop state authorizes reset, stash, destructive checkout, force-push, history rewrite,
automatic HF rollback, evidence modification, or expansion of the mutation boundary.

## 11. Credentials, privacy, and receipts

Hugging Face credentials may come only from the existing credential store or environment. They
must never be printed, logged, committed, serialized into evidence, or included in an environment
dump. Identity verification records only the authenticated username `steven0226`.

Receipts contain only public repository coordinates, revisions, filenames, sizes, ordinary blob
IDs, LFS SHA-256 values, README content hashes, Git SHAs, CI results, and release URLs. They contain
no credentials, private absolute paths, private notes, or machine-specific environment data.

## 12. Final acceptance record

Stable closure is accepted only when one public-safe record proves all of the following:

- final `main` SHA and tree;
- final package version `1.0.0`;
- evidence source commit `1a077a45e309594e5bb43743a8b84d89155595d4`;
- release pull request and successful PR/main/tag CI runs;
- annotated tag object and peeled commit;
- non-draft, non-prerelease source-only GitHub Release URL with zero additional assets;
- adapter and merged pre/post revisions and README identities;
- unchanged complete non-README inventories, including all LFS SHA-256 values and sizes;
- exact branch-protection settings;
- formal `kuotunyu` contributor/commit identities;
- zero open release pull requests and no temporary release branch/worktree; and
- clean canonical `main`.

The accepted terminal state is:

> `FROZEN / PORTFOLIO COMPLETE`

## 13. Current review gate

This document defines future execution but does not perform or authorize mutation within the
design-only commit that introduces it. No implementation plan, release-source change, push, pull
request, Hugging Face update, version change, tag, or Release proceeds until the owner completes
written review of this committed spec.
