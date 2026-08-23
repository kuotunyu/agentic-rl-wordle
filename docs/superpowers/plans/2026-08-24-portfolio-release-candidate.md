# Portfolio Release Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the local repository as a defensible `PORTFOLIO_RELEASE_CANDIDATE / EXTERNAL_PUBLICATION_REQUIRED` without changing training outputs, publishing externally, or overstating the 2.8% win rate.

**Architecture:** Treat packaging, evidence recomputation, provenance/public-boundary auditing, and release documentation as separate gates. Preserve `results/full_463_report.json` as the immutable committed aggregate, derive all statistical claims from it, and make clean Python 3.11/3.12 install/build paths executable in CI and disposable local environments.

**Tech Stack:** Python 3.11/3.12, setuptools, pytest, Ruff, PyPA build, GitHub Actions, Hugging Face public metadata (read-only audit).

## Global Constraints

- Do not download models; do not execute training, vLLM, GPU, or Colab workloads.
- Do not push or modify Hugging Face; do not create a GitHub repository, remote, tag, or release.
- Do not call paid APIs, touch another repository, or perform global cache/Docker cleanup.
- Do not alter raw evaluation/training outputs or statistics to improve performance.
- Keep the conclusion: protocol learning succeeded, strategy learning remained limited; 2.8% is not a practical solver.
- Use the formal identities `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` for Git and `steven0226` for Hugging Face.
- Work in the owner-specified `main` checkout; owner has explicitly approved release governance and minimal hardening on this branch.

---

### Task 1: Packaging and dependency contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/wordle_rl/__init__.py`
- Modify: `requirements.txt`
- Create: `constraints/dev.txt`

**Interfaces:**
- Consumes: existing zero-runtime-dependency core package and `src` layout.
- Produces: Python `>=3.11,<3.13`, version `1.0.0rc1`, formal author/URL metadata, `dev`/`play` extras, pinned build backend, and an exact local/CI constraints strategy.

- [x] **Step 1: Record the pre-install failure**

Run the declared package without an install and retain the already observed `ModuleNotFoundError` as the red reviewer-path evidence; do not add a permanent `PYTHONPATH` setting.

- [x] **Step 2: Normalize package metadata**

Set the distribution version through setuptools dynamic attr from `wordle_rl.__version__ = "1.0.0rc1"`; declare Apache-2.0 via `license`/`license-files`; add classifiers, keywords, formal author email, intended GitHub URL `https://github.com/kuotunyu/agentic-rl-wordle`, and both `steven0226` model URLs.

- [x] **Step 3: Define supported dependencies**

Keep runtime dependencies empty, keep heavyweight inference dependencies in `play`, and add `pytest`, `ruff`, and `build` to `dev`. Replace the ad hoc requirements file with `-c constraints/dev.txt` plus `-e .[dev]`.

- [x] **Step 4: Generate and inspect exact constraints**

Resolve the `dev` extra for Python 3.11 with exact versions in `constraints/dev.txt`; document that the file constrains local/CI tooling while `requirements-colab.txt` is historical GPU evidence and is not installed by the release gate.

- [x] **Step 5: Verify metadata**

Run `python -m build`, inspect wheel/sdist metadata, and confirm version, Python requirement, author, license, extras, and URLs are identical across artifacts.

### Task 2: Installed CLI and import smoke path

**Files:**
- Create: `tests/test_cli.py`
- Create: `src/wordle_rl/cli.py`
- Create: `src/wordle_rl/__main__.py`
- Modify: `play.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: the existing `play.py` behavior.
- Produces: `wordle-rl` and `python -m wordle_rl` entry points whose `--help` path requires no Torch/model/data download.

- [x] **Step 1: Write the failing CLI smoke test**

Add a subprocess test that runs `python -m wordle_rl --help` from a temporary working directory and asserts exit code 0 plus the `--answer` option.

- [x] **Step 2: Verify red**

Run `python -m pytest tests/test_cli.py -q` in an editable test environment and confirm failure because `wordle_rl.__main__` is absent.

- [x] **Step 3: Move the CLI into the package**

Move the current play implementation to `wordle_rl.cli:main`, add `wordle_rl.__main__`, register `[project.scripts] wordle-rl`, and leave `play.py` as a backward-compatible thin wrapper.

- [x] **Step 4: Verify green**

Run the focused CLI test, then import/CLI smoke from outside the repository using the built wheel.

### Task 3: Word-list provenance and split audit

**Files:**
- Create: `tests/test_fetch_words.py`
- Modify: `scripts/fetch_words.py`
- Create: `docs/data-governance.md`

**Interfaces:**
- Consumes: cfreshman answer/allowed gists and the MIT tabatkins fallback.
- Produces: revision-pinned URLs, normalized SHA-256 validation, immutable source metadata, and an explicit licensing/split audit.

- [x] **Step 1: Write the failing integrity test**

Mock same-count but altered downloads and assert `fetch_words.main()` rejects content that does not match the pinned normalized SHA-256.

- [x] **Step 2: Verify red**

Run `python -m pytest tests/test_fetch_words.py -q` and confirm the current count-only validation accepts the tampered content.

- [x] **Step 3: Pin revisions and hashes**

Pin answer gist revision `c46f451920d5cf6326d550fb2d6abb1642717852`, allowed gist revision `d7c9e02d45afd26e12a71b4564189a949c29e8a9`, and fallback revision `255b9469c4dad99a3b95cc4ddbe139b3d3747868`; validate normalized hashes `5209b35f823f8b80f0404f863bd80df06d6a966c6eb1016d69f38badc6eed5d0` and `99be2e38dadf3e26952af7cb4d963f65b632d5de91aa99e5ce308e4dc9617b65`.

- [x] **Step 4: Verify counts and split**

Fetch the small word lists, run tests proving 2,315 answers, 12,972 legal actions, 1,852/463 split, seed 42 determinism, complete coverage, and zero train/eval overlap.

- [x] **Step 5: Document licensing boundary**

State that cfreshman gists expose no explicit license, their word lists are fetched but not redistributed, the tabatkins fallback is MIT, and external publication must retain this caveat rather than treating “publicly accessible” as “openly licensed.”

### Task 4: Evidence recomputation and claim matrix

**Files:**
- Modify: `tests/test_full_463_analysis.py`
- Modify: `eval/analyze_full_463.py`
- Modify: `results/full_463_analysis.json`
- Modify: `results/full_463_analysis.md`
- Create: `docs/claim-matrix.md`

**Interfaces:**
- Consumes: immutable committed aggregate `results/full_463_report.json`.
- Produces: recomputed 0/463 and 13/463 counts, recovered 2,753-turn protocol/legal counts, Wilson intervals, exact paired McNemar, Bonferroni narrative, and headline-to-test traceability.

- [x] **Step 1: Write failing recomputation tests**

Assert Wilson intervals are recomputed from wins/n even when the input CI is tampered; assert the committed aggregate recovers tuned protocol `2749/2753`, legal actions `2748/2753`, and repeat `1/2753`.

- [x] **Step 2: Verify red**

Run the focused analysis tests and confirm failure because the current analyzer trusts `win_ci` and does not expose integer action counts.

- [x] **Step 3: Implement aggregate recovery and validation**

Recover common denominators from exact serialized rates with a maximum of `463*6`, validate all rates/counts, recompute Wilson with `wordle_rl.metrics.wilson_ci`, and retain the valid base-zero McNemar shortcut.

- [x] **Step 4: Regenerate committed analysis**

Run `python eval/analyze_full_463.py`; do not modify `results/full_463_report.json`; compare generated JSON/Markdown with tests.

- [x] **Step 5: Create claim matrix**

Map each README headline to `full_463_report.json`, `full_463_analysis.json`, the relevant recomputation function/test, evidence level (`aggregate` vs representative transcript), and the constrained interpretation.

### Task 5: Public-boundary and identity gate

**Files:**
- Create: `tests/test_publication_boundary.py`
- Create: `scripts/check_publication_boundary.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Git tracked files and built wheel/sdist archives.
- Produces: a deterministic nonzero-exit scan for token/private-key patterns, local user/Drive-specific paths, unapproved emails, tracked private notes, and notebook outputs.

- [x] **Step 1: Write failing scanner tests**

Assert a realistic `hf_...` token, private-key marker, `C:\\Users\\...` path, and notebook output are rejected while `HF_TOKEN`, `/content/drive/MyDrive/agentic-rl-wordle`, and the approved noreply identity are accepted.

- [x] **Step 2: Verify red**

Run the focused test and confirm failure because the scanner module is absent.

- [x] **Step 3: Implement scanner**

Scan tracked UTF-8 files plus zip-format wheel/sdist content, report file/member and finding category without echoing secret values, and reject tracked `interview.md`.

- [x] **Step 4: Verify repository and artifacts**

Run the scanner on `git ls-files` and `dist/`; confirm all notebook output arrays/execution counts are empty and only generic Drive/HF token references remain.

### Task 6: CI candidate and release documentation

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `docs/model_card.md`
- Create: `docs/huggingface-audit.md`
- Create: `docs/release-readiness.md`
- Create: `CHANGELOG.md`
- Create: `release/v1.0.0.md`

**Interfaces:**
- Consumes: packaging, evidence, provenance, HF revisions, and public-boundary gates.
- Produces: GitHub Actions candidate; local v1.0.0-rc.1 notes; explicit publication blockers; correct identities and evidence links.

- [x] **Step 1: Add CI quality/build/install gates**

Use Python 3.11/3.12, constraints installation, Ruff check/format, pinned word fetch, pytest without `PYTHONPATH`, package build, wheel and sdist clean installs, import/CLI smoke, evidence recomputation, and public-boundary scans.

- [x] **Step 2: Align README/model card wording**

Keep the 2.8% result explicitly non-practical; replace “public list” licensing implications; use clean-install commands; identify the intended GitHub URL without claiming it is already published.

- [x] **Step 3: Record HF audit**

Record adapter revision `ef1e98ce214921049b86dce7c104c88875130023`, merged revision `a59a4fb4c26e5d0612ce3a3574193ec58d46fc64`, adapter base model/PEFT metadata, stale remote model cards, and missing live GitHub linkage.

- [x] **Step 4: Write readiness and release notes**

Document local gates, exact blockers, deferred external actions, v1.0.0 candidate highlights, no-tag status, and recommended GitHub repository name `agentic-rl-wordle`.

### Task 7: Disposable installs, full verification, review, and candidate commit

**Files:**
- Modify only files required by verification findings.

**Interfaces:**
- Consumes: the complete candidate tree.
- Produces: clean editable/wheel/sdist evidence, full quality/test/build evidence, and a local candidate SHA.

- [x] **Step 1: Run editable clean environment**

Create a disposable Python 3.12 venv, install `-c constraints/dev.txt -e .[dev]`, fetch pinned words, run pytest without `PYTHONPATH`, import smoke, and both CLI help paths.

- [x] **Step 2: Run non-editable clean environments**

Build wheel/sdist, install each into separate disposable venvs, run isolated import/CLI smoke, and run the repository test suite against the wheel-installed package.

- [x] **Step 3: Run quality/security/evidence gates**

Run Ruff check/format, publication-boundary scan on repo and artifacts, analysis `--check`, metadata inspection, and `git diff --check`.

- [x] **Step 4: Review requirements line by line**

Compare the owner’s 12 requested outcomes and all hard constraints against `docs/release-readiness.md`; record evidence or blocker for every line.

- [x] **Step 5: Commit with approved identity**

Confirm author/committer config, commit all intended tracked changes with a release-hardening message, re-run verification on the committed tree, and report the resulting candidate SHA without tag/push/remote creation.
