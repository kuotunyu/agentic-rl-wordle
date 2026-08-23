# Release readiness — v1.0.0 candidate

Status: **`PORTFOLIO_RELEASE_CANDIDATE / EXTERNAL_PUBLICATION_REQUIRED`**
Candidate package version: `1.0.0rc1`
Final tag: **not created**

## Owner-requested closure matrix

| # | Requirement | Local candidate evidence | Status |
|---:|---|---|---|
| 1 | Disposable editable/non-editable installs | Fresh Python 3.12 editable, Python 3.11 regular, Python 3.12 wheel, and Python 3.12 sdist venvs passed install/import/CLI gates; editable, regular, and wheel paths each passed 135 tests without `PYTHONPATH` | Ready |
| 2 | Unified metadata/Python/dependencies | `pyproject.toml`: Python `>=3.11,<3.13`, dynamic `1.0.0rc1`, zero-dependency core, bounded `dev`/`play`, pinned setuptools/wheel | Ready |
| 3 | Reproducible lock/constraints | `constraints/dev.txt` exactly pins the local/CI tool graph; `requirements.txt` consumes it | Ready for local/CI; historical GPU stack is separate |
| 4 | Clean pytest/CLI/import | 135 tests pass after editable, regular, and wheel installs with no `PYTHONPATH`; non-editable tests set only documented `WORDLE_RL_DATA_DIR` for fetch-only data; isolated installed help does not import Torch or download data/models | Ready |
| 5 | Recompute 463 evidence | `eval/analyze_full_463.py`, schema-2 analysis, focused and committed-artifact tests | Ready at aggregate evidence level |
| 6 | Claim matrix | `docs/claim-matrix.md` | Ready |
| 7 | Split/seed/source/license audit | `docs/data-governance.md`; revision/hash fetch; executable split tests | Ready with explicit cfreshman license caveat |
| 8 | HF revisions/card/linkage | `docs/huggingface-audit.md` | Audited; external fixes blocked by no-push restriction |
| 9 | Transcripts/prompts/paths/tokens/private data | `scripts/check_publication_boundary.py`; notebooks have no outputs; `interview.md` is ignored; Drive paths are generic; only token variable names remain | Ready, with full raw-eval absence disclosed |
| 10 | Actions/Ruff/build/clean install/security | `.github/workflows/ci.yml`, Ruff config, build/install jobs, evidence and boundary gates; final local mirror passed Ruff, build, artifact scan, metadata comparison, and disposable installs | Ready locally; public Actions run awaits GitHub publication |
| 11 | Readiness/CHANGELOG/v1.0.0 data | This file, `CHANGELOG.md`, `release/v1.0.0.md`; no tag | Ready |
| 12 | Formal identities/URLs | package author/maintainer and every Git commit use `kuotunyu` noreply; HF URLs use `steven0226`; intended repo is `kuotunyu/agentic-rl-wordle` | Ready locally; GitHub URL not live |

## Claim posture

Approved wording: protocol learning succeeded; strategy learning remained limited. The tuned model
produced 13/463 wins (2.8%) and is **not** a practical Wordle solver. Exact paired significance
supports a nonzero task-success difference in this fixed paired evaluation; it does not turn the
small absolute win rate into practical capability or an independent replication.

The training-curve figures in historical docs lack committed raw `metrics.jsonl`. They are run
records, not release-time recomputable headline evidence. The primary portfolio headline relies on
the committed 463 aggregate and explicitly states its evidence level.

## External publication blockers

These cannot be closed without actions prohibited in the current task:

1. **No GitHub publication:** the repository has no remote; `kuotunyu/agentic-rl-wordle` must be
   created and this candidate SHA pushed before package URLs/model-card links become live.
2. **Stale Hugging Face cards:** both public cards still say 4/200 and “not significant.” Sync the
   local full-463 card to adapter and merged repositories without changing weights.
3. **Missing bidirectional evidence linkage:** both HF cards must link the immutable Git candidate
   SHA; GitHub release evidence must pin the two HF revisions recorded in the HF audit.
4. **Incomplete model lineage manifest:** adapter metadata does not pin the upstream Qwen revision,
   and no public manifest cryptographically links base + adapter → merged output.
5. **Evaluation provenance is not self-contained:** the committed aggregate does not embed Git SHA,
   prompt/bundle hash, or adapter revision. Current linkage can be audited historically but not
   proven from the result file alone.

Evidence/license decisions required at publication time:

- Either publish redacted full per-episode evaluation records and their hash, or retain the current
  aggregate-only evidence disclosure and avoid “raw-recomputed” wording.
- Keep cfreshman lists fetch-only and preserve the “no explicit license found” notice; Apache-2.0
  must not be presented as licensing those third-party lists.
- If exact historical training-environment recreation is a publication requirement, recover the
  original complete `pip freeze`/container manifest. `requirements-colab.txt` records direct
  constraints and observed core versions but is not a fully resolved GPU lock.

## Final external sequence

1. Create `kuotunyu/agentic-rl-wordle` and push the local candidate commit.
2. Let GitHub Actions pass on the public SHA.
3. Resolve the evidence/license decisions above and amend docs only by lowering unsupported claims.
4. Update both HF cards and add exact Git/HF/model-lineage links.
5. Re-run public-boundary and claim checks on the final tree.
6. Only then create the annotated `v1.0.0` tag and GitHub Release.

Recommended GitHub repository name: **`agentic-rl-wordle`**.
