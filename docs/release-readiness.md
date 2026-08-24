# Release readiness — v1.0.0

Status: **`HF_README_ONLY_UPDATE_VERIFIED`; Phase D exact-head PR CI is required before merge review**

Candidate package version: `1.0.0`

Final tag: **not created**

## Phase A release contract

| Requirement | Evidence | Required result |
|---|---|---|
| Stable package identity | `src/wordle_rl/__init__.py`, dynamic metadata, wheel and sdist metadata | Source, wheel, sdist, isolated imports, and both CLIs report `1.0.0` |
| Python reviewer paths | Disposable Python 3.11/3.12 editable and regular installs | Full pytest, isolated import, module CLI, and console CLI pass without `PYTHONPATH` |
| Quality and analysis | Ruff, `eval/analyze_full_463.py --check`, complete pytest | All pass; aggregate recomputation is exact |
| Immutable research evidence | Evidence commit `1a077a45e309594e5bb43743a8b84d89155595d4` and release-contract test | Research/evaluation blobs remain identical |
| Public boundary | `scripts/check_publication_boundary.py` and release-contract scans | No secret, credential, private path, notebook output, or oversized artifact finding |
| Claim discipline | `README.md`, both authoritative cards, `docs/claim-matrix.md` | Exact metrics and bounded conclusion; no practical-solver claim |
| HF closure | `docs/huggingface-audit.md` | Exact pre/post inventories; only the two authorized README blobs changed |
| Distribution artifacts | PyPA wheel and sdist | Exactly one of each; install and CLI smoke pass; no weights or word lists included |
| Git scope | Exact release-path allowlist and formal commit identity | No `pyproject.toml`, training, evaluation, reward, protocol, dataset, or result change |

The execution report, rather than this self-referential source file, records the exact Phase A
candidate SHA/tree, gate outputs, distribution hashes, and clean worktree result.

## Verified Hugging Face README-only transaction

- Adapter post revision: `e95fc44d5914d800483a847e8768b86f33719f12`
- Merged post revision: `94c0524cf963f7b22f1dc253eda5b4ef5a075956`
- Adapter README SHA-256: `ab9c473a8eb9efaf2ddc32873d405bb5eb6b5e305a0dddcdf774f8b7a77a0e6b`
- Merged README SHA-256: `b9f67c27188839000385ec85900a1d6825157aec1516af883349b7e52efb8e47`
- Public-safe receipt SHA-256:
  `c4e429d0c34ad32a515eaae541608107afaec1f9e4bc4afa16f24b9d9561bd2d`

Both filename sets remained at eight files. Every non-README blob, ordinary size, LFS SHA-256,
and LFS size remained unchanged. The authoritative pre/post inventory is recorded in
`docs/huggingface-audit.md`; no weight, tokenizer, configuration, or training artifact changed.

## Claim posture

Approved wording: **Protocol learning succeeded; strategy learning remained limited; the 2.81%
win rate is not a practical Wordle solver.**

The fixed paired aggregate contains base 0/463 and tuned 13/463 (2.81%), with tuned Wilson 95% CI
1.65%–4.74%, exact paired McNemar `p=0.000244140625`, and Bonferroni-adjusted
`p=0.00048828125`. Protocol adherence was 2749/2753 (99.85%) and legal actions were 2748/2753
(99.82%). These results support protocol learning and a small task-success difference, not
practical solving capability, complete strategy learning, production readiness, or general RL
superiority.

## Evidence and lineage limits

- Only aggregate 463-game evidence is committed; full per-episode records are unavailable.
- The historical GPU environment is not bit-for-bit reconstructable.
- The exact upstream Qwen commit, adapter-to-merged command/manifest, and complete
  run→code→prompt→bundle→model cryptographic chain were not preserved.
- The cfreshman word lists remain fetch-only with no explicit license; Apache-2.0 does not license
  those third-party lists.
- Representative transcripts are illustrations, not a complete raw corpus.

## Sequential external gates

1. **Phase B — complete:** the exact Phase A branch was pushed to one open PR and its six exact-head
   CI jobs succeeded.
2. **Phase C — complete and verified:** adapter-first README-only commits passed full pre/post
   inventory checks; no weight or non-README artifact changed.
3. **Phase D — current authorized checkpoint:** record these immutable receipts in the same PR,
   push one closure commit, and require all six exact PR-head jobs again. Do not merge.
4. **Phase E — new authorization required:** fast-forward-only main, final main CI, annotated tag,
   tag CI, source-only GitHub Release with zero additional assets, protection verification, and
   scoped cleanup.

Phase E remains separately authorization-gated. No completed checkpoint supplies authority for
merge, main, tag, Release, or cleanup.
