# Release readiness — v1.0.0

Status: **`LOCAL_RELEASE_CANDIDATE_READY` after the Phase A execution receipt is complete**

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
| HF pre-state | `docs/huggingface-audit.md` | Complete approved inventories; no invented post-update revision |
| Distribution artifacts | PyPA wheel and sdist | Exactly one of each; install and CLI smoke pass; no weights or word lists included |
| Git scope | Exact release-path allowlist and formal commit identity | No `pyproject.toml`, training, evaluation, reward, protocol, dataset, or result change |

The execution report, rather than this self-referential source file, records the exact Phase A
candidate SHA/tree, gate outputs, distribution hashes, and clean worktree result.

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

1. **Phase B — authorization required:** push only the exact Phase A branch, create one PR, and
   require all six exact PR-head CI jobs to succeed. Keep the PR open.
2. **Phase C — separate authorization required:** revalidate account and full HF inventories, then
   perform adapter-first, README-only optimistic commits. No weight or non-README mutation.
3. **Phase D — same authorized PR only:** record exact HF post-update revisions and receipts in
   Git, push the closure commit, and require all six exact PR-head jobs again. Do not merge.
4. **Phase E — new authorization required:** fast-forward-only main, final main CI, annotated tag,
   tag CI, source-only GitHub Release with zero additional assets, protection verification, and
   scoped cleanup.

No Phase A authority permits any external mutation. No later checkpoint inherits authority from
an earlier checkpoint.
