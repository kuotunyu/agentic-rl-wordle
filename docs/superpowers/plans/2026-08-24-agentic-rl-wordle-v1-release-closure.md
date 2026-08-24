# Agentic RL Wordle v1.0.0 Release Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the approved `agentic-rl-wordle` source and two README-only Hugging Face model-card updates as an identity-preserving `v1.0.0` portfolio release without changing research evidence or model artifacts.

**Architecture:** Treat local release preparation, public PR evidence, the two-repository Hugging Face transaction, immutable receipt closure, and stable GitHub publication as five sequential fail-closed phases. The immutable research/evidence source remains `1a077a45e309594e5bb43743a8b84d89155595d4`; the release branch descends from protected `main` at `8ce548b7b7ae2b812dbacadf477b6600e9d2d867`. Hugging Face cards link the existing evidence commit and future stable URL, then a later Git commit records the resulting HF revisions before `main`, tag, and Release closure.

**Tech Stack:** Python 3.11/3.12, pytest, Ruff, PyPA build, setuptools, PowerShell, Git, GitHub CLI, GitHub Actions, `huggingface_hub`, and Hugging Face model metadata APIs.

## Global Constraints

- This committed plan is not execution authorization. Before Task 1, obtain a later written owner
  execution receipt naming the exact reviewed plan path and then-current branch HEAD; without it,
  remain at `IMPLEMENTATION_PLAN_COMMITTED / NO_IMPLEMENTATION_STARTED`.
- Execute only from branch `codex/v1.0.0-release-closure` in the owner-designated isolated worktree.
- Treat `docs/superpowers/specs/2026-08-24-agentic-rl-wordle-v1-release-closure-design.md` as normative.
- Treat `1a077a45e309594e5bb43743a8b84d89155595d4` as the immutable research/evidence source, not the final release commit.
- Preserve `results/full_463_report.json`, `results/full_463_analysis.json`, all baseline artifacts, notebooks, datasets, word lists, training/reward/protocol/evaluation logic, and historical commits byte-for-byte.
- Keep `pyproject.toml` unchanged; its dynamic version, metadata, dependency, Python-range, author,
  license, and URL contract are verified from the built distributions rather than edited.
- Keep the only approved conclusion: “Protocol learning succeeded; strategy learning remained limited; the 2.81% win rate is not a practical Wordle solver.”
- Publish no PyPI distribution, Docker image, Space, dataset, new model repository, model binary, or extra GitHub Release asset.
- Never execute training, GPU evaluation, vLLM, or Colab workloads, and never download model weights.
- Restrict each Hugging Face repository mutation to one `README.md`-only commit with optimistic concurrency against the approved parent revision.
- Obtain credentials only from the existing credential store or environment. Never print, record, commit, or include credential material in a receipt.
- Use `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` for every Git author, committer, and annotated tag identity.
- Use exact action pins:
  - `actions/checkout@11d5960a326750d5838078e36cf38b85af677262`
  - `actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065`
  - `actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02`
- A canceled, timed-out, skipped-required, neutral, or allowed-failure job is failure. The six required job names must each be `completed/success` on the exact PR head, final `main`, and tag commit.
- Stop immediately on any mismatch or forbidden mutation. Do not reset, stash, rebase, cherry-pick, force-push, rewrite history, or automatically roll back a successful HF README commit.

## File and interface map

| Path | Responsibility |
|---|---|
| `src/wordle_rl/__init__.py` | Single source of distribution version `1.0.0`. |
| `.github/workflows/ci.yml` | Exact action pins, version assertions, artifact name, and six public gates. |
| `tests/test_release_closure.py` | Offline, deterministic release-contract tests and immutable-evidence guard. |
| `docs/model_card.md` | Authoritative UTF-8/LF LoRA adapter card bytes. |
| `docs/model_card_merged.md` | Authoritative UTF-8/LF merged full-model card bytes. |
| `README.md` | Public project summary, exact claims, evidence links, and limitations. |
| `docs/claim-matrix.md` | Claim → aggregate artifact → recomputation test → evidence boundary. |
| `docs/huggingface-audit.md` | Approved HF pre-state and later exact post-state receipts. |
| `docs/release-readiness.md` | Sequential gate status and stop conditions. |
| `CHANGELOG.md` | Stable source-release and card-only publication history. |
| `release/v1.0.0.md` | Committed invariant GitHub Release body; Phase E prepends the then-known final SHA in a public-safe temporary body to avoid self-reference. |
| System temporary receipt JSON | Public-safe Phase C pre/post inventories; never committed and never contains credentials or local paths. |

### Shared test interfaces

`tests/test_release_closure.py` incrementally defines and reuses:

```python
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from scripts.check_publication_boundary import scan_text

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_SHA = "1a077a45e309594e5bb43743a8b84d89155595d4"
ADAPTER_CARD = ROOT / "docs" / "model_card.md"
MERGED_CARD = ROOT / "docs" / "model_card_merged.md"


def read_text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def git_text(revision: str, relative: str) -> str:
    return subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_backticked_value(label: str, text: str) -> str:
    match = re.search(rf"^{re.escape(label)}: `([^`]+)`$", text, flags=re.MULTILINE)
    assert match is not None, label
    return match.group(1)
```

Later tasks must keep these names and signatures unchanged.

### Shared GitHub verification interfaces

Every task that validates PR, `main`, or tag CI must define and call this PowerShell function in
the same shell session. It rejects missing, duplicate, added, skipped, canceled, timed-out,
neutral, or otherwise non-successful jobs:

```powershell
function Assert-SixSuccessJobs {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory = $true)] [object] $RunView,
    [Parameter(Mandatory = $true)] [string] $ExpectedSha
  )

  $Required = @(
    "quality",
    "build-artifacts",
    "test-install (3.11, editable)",
    "test-install (3.11, regular)",
    "test-install (3.12, editable)",
    "test-install (3.12, regular)"
  )
  $Actual = @($RunView.jobs | ForEach-Object { $_.name })
  if ($Actual.Count -ne 6 -or
      @(Compare-Object ($Required | Sort-Object) ($Actual | Sort-Object)).Count -ne 0) {
    throw "Required job set mismatch"
  }
  if ($RunView.headSha -ne $ExpectedSha -or
      $RunView.status -ne "completed" -or
      $RunView.conclusion -ne "success") {
    throw "Workflow run is not completed/success for the expected SHA"
  }
  if (@($RunView.jobs | Where-Object {
        $_.status -ne "completed" -or $_.conclusion -ne "success"
      }).Count -ne 0) {
    throw "At least one required job is not completed/success"
  }
}

function Test-SixSuccessValidatorFailsClosed {
  $Jobs = @(
    "quality",
    "build-artifacts",
    "test-install (3.11, editable)",
    "test-install (3.11, regular)",
    "test-install (3.12, editable)",
    "test-install (3.12, regular)"
  ) | ForEach-Object {
    [pscustomobject]@{ name = $_; status = "completed"; conclusion = "success" }
  }
  $Fixture = [pscustomobject]@{
    headSha = "0000000000000000000000000000000000000000"
    status = "completed"
    conclusion = "success"
    jobs = $Jobs
    url = "fixture://six-success"
  }
  Assert-SixSuccessJobs -RunView $Fixture -ExpectedSha $Fixture.headSha
  $Fixture.jobs[0].conclusion = "failure"
  $Rejected = $false
  try {
    Assert-SixSuccessJobs -RunView $Fixture -ExpectedSha $Fixture.headSha
  } catch {
    $Rejected = $true
  }
  if (-not $Rejected) { throw "CI validator accepted a failed job" }
}
```

Every task that verifies `main` protection must call this exact interface on a fresh API result:

```powershell
function Assert-MainProtection {
  [CmdletBinding()]
  param([Parameter(Mandatory = $true)] [object] $Protection)

  $Required = @(
    "quality",
    "build-artifacts",
    "test-install (3.11, editable)",
    "test-install (3.11, regular)",
    "test-install (3.12, editable)",
    "test-install (3.12, regular)"
  )
  $Contexts = @($Protection.required_status_checks.contexts)
  if (-not $Protection.required_status_checks.strict -or
      $Contexts.Count -ne 6 -or
      @(Compare-Object ($Required | Sort-Object) ($Contexts | Sort-Object)).Count -ne 0) {
    throw "Required-check protection mismatch"
  }
  if (-not $Protection.enforce_admins.enabled -or
      -not $Protection.required_linear_history.enabled -or
      $Protection.allow_force_pushes.enabled -or
      $Protection.allow_deletions.enabled) {
    throw "Main protection policy mismatch"
  }
}
```

---

# Phase A — Local release candidate

No GitHub or Hugging Face mutation is allowed in Phase A. Its terminal state is
`LOCAL_RELEASE_CANDIDATE_READY`.

### Task 1: Lock stable version, action pins, and immutable evidence

**Files:**
- Create: `tests/test_release_closure.py`
- Modify: `src/wordle_rl/__init__.py:7`
- Modify: `.github/workflows/ci.yml:16-97`

**Interfaces:**
- Consumes: dynamic package metadata from `wordle_rl.__version__`, approved action SHAs, evidence source SHA.
- Produces: stable source/workflow version `1.0.0`, exact pinned actions, artifact name `wordle-rl-1.0.0`, and `test_research_artifacts_match_evidence_source()`.

- [ ] **Step 1: Add the shared helpers and failing stable-release tests**

Use `apply_patch` to create `tests/test_release_closure.py` with the shared interfaces above and:

```python
IMMUTABLE_PATHS = (
    "results/full_463_report.json",
    "results/full_463_report.md",
    "results/full_463_analysis.json",
    "results/full_463_analysis.md",
    "results/baselines.json",
    "results/baselines.md",
    "results/final_report.md",
    "baselines/run_baseline.py",
    "wordle_rl_bundle.sha256",
    "wordle_grpo_colab_train.ipynb",
    "wordle_full463_eval_colab.ipynb",
    "wordle_eval_push_colab.ipynb",
    "data/.gitkeep",
    "scripts/fetch_words.py",
    "src/wordle_rl/config.py",
    "src/wordle_rl/metrics.py",
    "src/wordle_rl/rewards.py",
    "src/wordle_rl/protocol.py",
    "src/wordle_rl/rollout.py",
    "src/wordle_rl/train.py",
    "src/wordle_rl/words.py",
    "eval/analyze_full_463.py",
    "eval/run_eval.py",
)

CHECKOUT_SHA = "11d5960a326750d5838078e36cf38b85af677262"
SETUP_PYTHON_SHA = "a26af69be951a213d495a4c3e4e4022e16d87065"
UPLOAD_ARTIFACT_SHA = "ea165f8d65b6e75b540449e92b4886f43607fa02"


def test_source_and_workflow_use_stable_version_and_exact_action_pins():
    source = read_text("src/wordle_rl/__init__.py")
    workflow = read_text(".github/workflows/ci.yml")

    assert '__version__ = "1.0.0"' in source
    assert "1.0.0rc1" not in workflow
    assert workflow.count(f"actions/checkout@{CHECKOUT_SHA}") == 3
    assert workflow.count(f"actions/setup-python@{SETUP_PYTHON_SHA}") == 3
    assert workflow.count(f"actions/upload-artifact@{UPLOAD_ARTIFACT_SHA}") == 1
    assert "actions/checkout@v4" not in workflow
    assert "actions/setup-python@v5" not in workflow
    assert "actions/upload-artifact@v4" not in workflow
    assert "name: wordle-rl-1.0.0" in workflow


def test_research_artifacts_match_evidence_source():
    for relative in IMMUTABLE_PATHS:
        assert (ROOT / relative).read_bytes() == subprocess.run(
            ["git", "show", f"{EVIDENCE_SHA}:{relative}"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout, relative
```

- [ ] **Step 2: Run RED and record the intended failures**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
py -3.12 -m pytest -p no:cacheprovider tests/test_release_closure.py -q
```

Expected: `test_source_and_workflow_use_stable_version_and_exact_action_pins` fails because the
source/workflow still contain `1.0.0rc1` and moving action tags. The immutable-artifact test passes.

- [ ] **Step 3: Apply the minimal stable-version and workflow changes**

Use `apply_patch` for these exact replacements:

```python
__version__ = "1.0.0"
```

```yaml
- uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262
- uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065
- uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02
```

Replace all four CI version assertions with `1.0.0` and set the uploaded artifact name to
`wordle-rl-1.0.0`. Do not change job names, triggers, permissions, matrices, commands, or workflow
semantics.

- [ ] **Step 4: Run GREEN and the workflow regression test**

Run:

```powershell
py -3.12 -m pytest -p no:cacheprovider tests/test_release_closure.py tests/test_publication_boundary.py::test_quality_workflow_tests_merge_result_but_scans_event_publication_tip -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Verify scope and commit**

Run:

```powershell
git diff --check
git diff --name-only
git diff --exit-code 1a077a45e309594e5bb43743a8b84d89155595d4 -- results baselines wordle_grpo_colab_train.ipynb wordle_full463_eval_colab.ipynb wordle_eval_push_colab.ipynb wordle_rl_bundle.sha256 src/wordle_rl/rewards.py src/wordle_rl/protocol.py eval
git add src/wordle_rl/__init__.py .github/workflows/ci.yml tests/test_release_closure.py
git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit -m "release: finalize v1.0.0 package and CI contract"
```

Expected: the immutable diff is empty and the commit changes only the three declared paths.

### Task 2: Create distinct authoritative adapter and merged cards

**Files:**
- Modify: `tests/test_release_closure.py`
- Modify: `docs/model_card.md`
- Create: `docs/model_card_merged.md`

**Interfaces:**
- Consumes: approved metrics, evidence/source URL, stable release URL, aggregate and lineage boundaries.
- Produces: distinct canonical UTF-8/LF payloads whose Git blob bytes are the only allowed HF README payloads.

- [ ] **Step 1: Add failing card-role and claim tests**

Append with `apply_patch`:

```python
SHARED_CARD_LITERALS = (
    "0/463",
    "13/463 = 2.81%",
    "0.00%–0.82%",
    "1.65%–4.74%",
    "2749/2753 = 99.85%",
    "2748/2753 = 99.82%",
    "0.000244140625",
    "0.00048828125",
    "1340/2290",
    "1119/2290",
    "Protocol learning succeeded; strategy learning remained limited",
    "not a practical Wordle solver",
    "full per-episode records are unavailable",
    "documentary lineage, not complete cryptographic proof",
    "cfreshman word lists are fetch-only and have no explicit license",
    "https://github.com/kuotunyu/agentic-rl-wordle/commit/1a077a45e309594e5bb43743a8b84d89155595d4",
    "https://github.com/kuotunyu/agentic-rl-wordle/releases/tag/v1.0.0",
)


def test_authoritative_cards_are_distinct_and_honest():
    adapter = ADAPTER_CARD.read_text(encoding="utf-8")
    merged = MERGED_CARD.read_text(encoding="utf-8")

    assert adapter != merged
    assert "LoRA adapter" in adapter
    assert "Qwen/Qwen2.5-1.5B-Instruct" in adapter
    assert "rank 16" in adapter
    assert "alpha 32" in adapter
    assert "dropout 0.05" in adapter
    assert "must be loaded together with the compatible base model" in adapter
    assert "reported tuned evaluation is the adapter evaluation" in adapter

    assert "merged full model" in merged
    assert "does not require a separately attached LoRA adapter" in merged
    assert "No independent 463-word evaluation was run against these merged bytes" in merged
    assert "not an independent replication" in merged

    for literal in SHARED_CARD_LITERALS:
        assert literal in adapter, literal
        assert literal in merged, literal

    assert adapter.startswith("---\nlicense: apache-2.0\n")
    assert merged.startswith("---\nlicense: apache-2.0\n")
    assert "library_name: peft" in adapter
    assert "library_name: peft" not in merged
    assert "Apache-2.0 does not license the cfreshman word lists" in adapter
    assert "Apache-2.0 does not license the cfreshman word lists" in merged


def test_authoritative_cards_are_public_boundary_clean():
    for path in (ADAPTER_CARD, MERGED_CARD):
        assert scan_text(path.name, path.read_text(encoding="utf-8")) == []
```

- [ ] **Step 2: Run RED**

Run:

```powershell
py -3.12 -m pytest -p no:cacheprovider tests/test_release_closure.py::test_authoritative_cards_are_distinct_and_honest tests/test_release_closure.py::test_authoritative_cards_are_public_boundary_clean -q
```

Expected: collection/test failure because `docs/model_card_merged.md` does not exist and the
adapter card lacks the exact approved wording.

- [ ] **Step 3: Write the authoritative adapter card**

Use `apply_patch` to rewrite `docs/model_card.md`. Its front matter begins exactly:

```yaml
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
```

The body contains sections `Repository role`, `Evaluation`, `Evidence links`, `Limitations`, and
`Loading`. Use the exact shared literals from the test. State that it is a LoRA adapter with rank
16, alpha 32, dropout 0.05, that it must be loaded with the compatible base model, and that the
reported tuned result is the adapter evaluation.

- [ ] **Step 4: Write the authoritative merged card**

Use `apply_patch` to create `docs/model_card_merged.md`. Its front matter begins exactly:

```yaml
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
```

Use the same section names and exact shared literals, but identify merged full-model weights,
state that a separate adapter attachment is unnecessary, state that no independent 463-word
evaluation was run against the merged bytes, and prohibit treating the adapter result as an
independent merged-model replication.

Both cards state all four evidence limits: aggregate-only records, incomplete historical GPU
environment, incomplete upstream/adapter→merged cryptographic lineage, and unlicensed fetch-only
cfreshman word lists.

- [ ] **Step 5: Run GREEN and byte-format gates**

Run:

```powershell
py -3.12 -m pytest -p no:cacheprovider tests/test_release_closure.py -q
py -3.12 scripts/check_publication_boundary.py docs/model_card.md docs/model_card_merged.md --identity-tip HEAD
git diff --check
```

Expected: all tests/scans pass; both files are UTF-8, contain LF in the staged Git blobs, and are
not byte-identical to each other.

- [ ] **Step 6: Commit the immutable card payloads**

Run:

```powershell
git add tests/test_release_closure.py docs/model_card.md docs/model_card_merged.md
git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit -m "docs: add distinct Hugging Face model cards"
```

Expected: one scoped commit. No later Phase A task changes either card.

### Task 3: Align README and claim matrix with exact evidence

**Files:**
- Modify: `tests/test_release_closure.py`
- Modify: `README.md`
- Modify: `docs/claim-matrix.md`

**Interfaces:**
- Consumes: authoritative aggregate values and evidence limits.
- Produces: exact public claims with no stale “GitHub not created” state and no capability expansion.

- [ ] **Step 1: Add failing public-claim tests**

Append:

```python
PUBLIC_CLAIM_LITERALS = (
    "0/463",
    "13/463 (2.81%)",
    "99.85% protocol adherence",
    "99.82% legal actions",
    "0.000244140625",
    "0.00048828125",
    "not a practical Wordle solver",
    "aggregate-only",
)


def test_readme_and_claim_matrix_use_exact_bounded_claims():
    readme = read_text("README.md")
    matrix = read_text("docs/claim-matrix.md")

    for literal in PUBLIC_CLAIM_LITERALS:
        assert literal in readme, literal
        assert literal in matrix, literal

    assert "GitHub repo 尚未建立" not in readme
    assert "full raw-record recomputation" not in readme
    assert "practical or strong Wordle solver" not in readme
    assert "1a077a45e309594e5bb43743a8b84d89155595d4" in readme
    assert "immutable research/evidence source commit" in matrix
    assert "not the final release commit" in matrix
```

- [ ] **Step 2: Run RED**

Run:

```powershell
py -3.12 -m pytest -p no:cacheprovider tests/test_release_closure.py::test_readme_and_claim_matrix_use_exact_bounded_claims -q
```

Expected: FAIL because the README still contains stale publication status and rounded wording.

- [ ] **Step 3: Apply exact README and matrix updates**

Use `apply_patch`. Preserve the project narrative and examples, but replace public headline/table
display values with the tested literals. Add the evidence source link and explicitly distinguish
it from the later final release commit. Preserve the sentence that protocol learning succeeded,
strategy learning remained limited, and 2.81% is not practical capability.

Do not add new performance claims, transcripts, results, model lineage, or raw-evidence claims.

- [ ] **Step 4: Run GREEN and aggregate recomputation**

Run:

```powershell
py -3.12 -m pytest -p no:cacheprovider tests/test_release_closure.py tests/test_full_463_analysis.py -q
py -3.12 eval/analyze_full_463.py --check
```

Expected: tests pass and committed aggregate recomputation reports an exact match.

- [ ] **Step 5: Commit**

Run:

```powershell
git diff --check
git add tests/test_release_closure.py README.md docs/claim-matrix.md
git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit -m "docs: align stable release claims"
```

### Task 4: Prepare pre-HF audit and stable release documentation

**Files:**
- Modify: `tests/test_release_closure.py`
- Modify: `docs/huggingface-audit.md`
- Modify: `docs/release-readiness.md`
- Modify: `CHANGELOG.md`
- Modify: `release/v1.0.0.md`

**Interfaces:**
- Consumes: approved pre-update HF revisions/inventories and stable source state.
- Produces: a truthful pre-mutation release-source candidate with state `HF README-only update pending owner authorization`; it contains no invented post-update revision.

- [ ] **Step 1: Add failing pre-HF evidence tests**

Append:

```python
PRE_HF_LITERALS = (
    "ef1e98ce214921049b86dce7c104c88875130023",
    "a59a4fb4c26e5d0612ce3a3574193ec58d46fc64",
    "c3ab2ecbc0a032e77345239b02f41b967a8398017e312d7a2ea8e45a04afcf5b",
    "d3a35ef0db5324b3f67135e0cd216dbd980cd6afc0cf03aaf6621e36b9777e00",
    "92e6379ed7ddf363e7f500b143afa7a2dc725d3e86bd87bc9eb933831c7d68b7",
    "9df95d5c562cd327397015f3324e3627fd280725af4f10be014233835e778ab8",
    "b6c55086e798e1f62e6d970f07ee97ab39c1e0af3ee4b6ecdb2a349e485087af",
)


def test_pre_hf_release_documents_exact_baseline_without_fake_post_state():
    audit = read_text("docs/huggingface-audit.md")
    readiness = read_text("docs/release-readiness.md")
    changelog = read_text("CHANGELOG.md")
    notes = read_text("release/v1.0.0.md")

    for literal in PRE_HF_LITERALS:
        assert literal in audit, literal

    assert "HF README-only update pending owner authorization" in audit
    assert "LOCAL_RELEASE_CANDIDATE_READY" in readiness
    assert "1.0.0rc1" not in readiness
    assert "## [1.0.0] - 2026-08-24" in changelog
    assert "source-only" in notes
    assert "zero additional assets" in notes
    assert EVIDENCE_SHA in notes
    assert "exact upstream Qwen commit was not preserved" in audit
    assert "adapter-to-merged derivation is documentary lineage" in audit
    assert "Post-update adapter revision:" not in audit
    assert "Post-update merged revision:" not in audit
```

- [ ] **Step 2: Run RED**

Run:

```powershell
py -3.12 -m pytest -p no:cacheprovider tests/test_release_closure.py::test_pre_hf_release_documents_exact_baseline_without_fake_post_state -q
```

Expected: FAIL because the current docs still describe the GitHub repository as unpublished and
do not contain the complete approved inventory contract.

- [ ] **Step 3: Update the four release documents**

Use `apply_patch` only. `docs/huggingface-audit.md` records every filename/blob/LFS identity from
design spec Section 6, the two README hashes, the three immutable binary identities, and the exact
pending-state sentence. `docs/release-readiness.md` records Phase A as ready while Phases B–E
remain authorization-gated. `CHANGELOG.md` creates the stable `1.0.0` entry and preserves that no
research evidence changed. `release/v1.0.0.md` contains the final approved claims and limits but
states that HF post-update receipts are not yet present because mutation has not occurred.

- [ ] **Step 4: Run GREEN and publication checks**

Run:

```powershell
py -3.12 -m pytest -p no:cacheprovider tests/test_release_closure.py -q
py -3.12 scripts/check_publication_boundary.py --identity-tip HEAD
git diff --check
```

Expected: all gates pass and the scanner emits no credential/private-path finding.

- [ ] **Step 5: Commit the release-source candidate docs**

Run:

```powershell
git add tests/test_release_closure.py docs/huggingface-audit.md docs/release-readiness.md CHANGELOG.md release/v1.0.0.md
git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit -m "docs: prepare v1.0.0 publication evidence"
```

### Task 5: Execute complete local release gates

**Files:**
- Verify only; no tracked file may change.

**Interfaces:**
- Consumes: completed Phase A tree.
- Produces: exact candidate SHA/tree, build hashes, full gate receipts, and terminal state `LOCAL_RELEASE_CANDIDATE_READY`.

- [ ] **Step 1: Verify branch shape, commit identities, and research bytes**

Run:

```powershell
$ReleaseHead = git rev-parse HEAD
git merge-base --is-ancestor 8ce548b7b7ae2b812dbacadf477b6600e9d2d867 $ReleaseHead
if ($LASTEXITCODE -ne 0) { throw "NON_FAST_FORWARD_CANDIDATE" }
if (@(git rev-list --merges 8ce548b7b7ae2b812dbacadf477b6600e9d2d867..$ReleaseHead).Count -ne 0) { throw "NON_LINEAR_CANDIDATE" }
py -3.12 scripts/check_publication_boundary.py --identity-tip $ReleaseHead
git diff --exit-code 1a077a45e309594e5bb43743a8b84d89155595d4 -- results baselines wordle_grpo_colab_train.ipynb wordle_full463_eval_colab.ipynb wordle_eval_push_colab.ipynb wordle_rl_bundle.sha256 src/wordle_rl/rewards.py src/wordle_rl/protocol.py eval
```

Expected: all commands succeed and both diffs are empty.

- [ ] **Step 2: Run quality, analysis, boundary, and full tests**

Run:

```powershell
ruff check .
ruff format --check .
py -3.12 eval/analyze_full_463.py --check
py -3.12 scripts/check_publication_boundary.py --identity-tip $ReleaseHead
py -3.12 -m pytest -p no:cacheprovider -q
git diff --check
```

Expected: Ruff and analysis pass, publication scan is clean, and all collected tests complete with
only the documented Linux/Windows conditional skip.

- [ ] **Step 3: Run Python 3.11/3.12 editable and regular install gates**

Run from the repository root:

```powershell
$GateRoot = Join-Path ([IO.Path]::GetTempPath()) "agentic-rl-wordle-v1-gates"
if (Test-Path -LiteralPath $GateRoot) { throw "Gate directory already exists" }
New-Item -ItemType Directory -Path $GateRoot | Out-Null

foreach ($Version in @("3.11", "3.12")) {
  foreach ($Mode in @("editable", "regular")) {
    $Venv = Join-Path $GateRoot ("py" + $Version.Replace(".", "") + "-" + $Mode)
    py -$Version -m venv $Venv
    $Python = Join-Path $Venv "Scripts/python.exe"
    if ($Mode -eq "editable") {
      & $Python -m pip install --disable-pip-version-check -c constraints/dev.txt -e ".[dev]"
    } else {
      & $Python -m pip install --disable-pip-version-check -c constraints/dev.txt ".[dev]"
    }
    & $Python -m pytest -p no:cacheprovider -q
    & $Python -I -c "import wordle_rl; assert wordle_rl.__version__ == '1.0.0'"
    & $Python -m wordle_rl --help
    & (Join-Path $Venv "Scripts/wordle-rl.exe") --help
  }
}
```

Expected: four environments pass install, full tests, isolated import, and both CLI paths without
`PYTHONPATH`. Do not run `scripts/fetch_words.py`; Phase A performs no GitHub or Hugging Face
network access.

- [ ] **Step 4: Build and smoke-test wheel and sdist**

Run:

```powershell
$BuildVenv = Join-Path $GateRoot "build"
py -3.12 -m venv $BuildVenv
$BuildPython = Join-Path $BuildVenv "Scripts/python.exe"
& $BuildPython -m pip install --disable-pip-version-check -c constraints/dev.txt build
$Dist = Join-Path $GateRoot "dist"
& $BuildPython -m build --outdir $Dist
py -3.12 scripts/check_publication_boundary.py $Dist --identity-tip $ReleaseHead

foreach ($Kind in @("wheel", "sdist")) {
  $Venv = Join-Path $GateRoot $Kind
  py -3.12 -m venv $Venv
  $Python = Join-Path $Venv "Scripts/python.exe"
  $Artifact = if ($Kind -eq "wheel") {
    Get-ChildItem -LiteralPath $Dist -Filter "*.whl" | Select-Object -ExpandProperty FullName
  } else {
    Get-ChildItem -LiteralPath $Dist -Filter "*.tar.gz" | Select-Object -ExpandProperty FullName
  }
  if (@($Artifact).Count -ne 1) { throw "Expected exactly one $Kind artifact" }
  & $Python -m pip install --disable-pip-version-check $Artifact
  Push-Location $GateRoot
  try {
    & $Python -I -c "import wordle_rl; assert wordle_rl.__version__ == '1.0.0'"
    & (Join-Path $Venv "Scripts/wordle-rl.exe") --help
  } finally {
    Pop-Location
  }
}
```

Expected: one wheel and one sdist, both install and report `1.0.0`; no model is downloaded.

- [ ] **Step 5: Verify metadata, exact file scope, and clean state**

Run:

```powershell
py -3.12 -c "import pathlib,tarfile,zipfile; d=pathlib.Path(r'$Dist'); w=next(d.glob('*.whl')); s=next(d.glob('*.tar.gz')); zw=zipfile.ZipFile(w); wm=next(n for n in zw.namelist() if n.endswith('METADATA')); wt=zw.read(wm).decode(); ts=tarfile.open(s); sm=next(n for n in ts.getnames() if n.endswith('PKG-INFO')); st=ts.extractfile(sm).read().decode(); assert 'Version: 1.0.0' in wt and 'Version: 1.0.0' in st; print('wheel/sdist metadata version: 1.0.0')"
$AllowedPaths = @(
  ".github/workflows/ci.yml",
  "CHANGELOG.md",
  "README.md",
  "docs/claim-matrix.md",
  "docs/huggingface-audit.md",
  "docs/model_card.md",
  "docs/model_card_merged.md",
  "docs/release-readiness.md",
  "docs/superpowers/plans/2026-08-24-agentic-rl-wordle-v1-release-closure.md",
  "docs/superpowers/specs/2026-08-24-agentic-rl-wordle-v1-release-closure-design.md",
  "release/v1.0.0.md",
  "src/wordle_rl/__init__.py",
  "tests/test_release_closure.py"
)
$ChangedPaths = @(git diff --name-only 8ce548b7b7ae2b812dbacadf477b6600e9d2d867..HEAD)
if (@(Compare-Object ($AllowedPaths | Sort-Object) ($ChangedPaths | Sort-Object)).Count -ne 0) { throw "Release path scope mismatch" }
git status --short
```

Expected: metadata assertion passes, the diff contains exactly the approved Git paths, and Git
status is empty.

- [ ] **Step 6: Remove only the verified disposable gate directory**

Run:

```powershell
$ResolvedGateRoot = (Resolve-Path -LiteralPath $GateRoot).Path
$ResolvedTemp = (Resolve-Path -LiteralPath ([IO.Path]::GetTempPath())).Path
if ((Split-Path -Leaf $ResolvedGateRoot) -ne "agentic-rl-wordle-v1-gates" -or
    -not $ResolvedGateRoot.StartsWith(($ResolvedTemp.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar), [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe cleanup target" }
Remove-Item -LiteralPath $ResolvedGateRoot -Recurse -Force
```

Expected: only the explicitly verified disposable gate directory is removed.

- [ ] **Step 7: Record the Phase A checkpoint**

Record candidate SHA/tree, all local gate results, immutable research diff result, card SHA-256
values, and clean status in the execution report. Stop at `LOCAL_RELEASE_CANDIDATE_READY` and wait
for the Phase B owner authorization receipt.

---

# Phase B — GitHub PR evidence

Phase B requires a new owner authorization envelope explicitly permitting branch pushes through
Phase D and creation/update of one open PR, while prohibiting merge, tag, and Release. Its terminal
state is `PR_OPEN_CI_GREEN_HF_UPDATE_REQUIRED`.

### Task 6: Obtain authorization and publish the exact release candidate PR

**Files:**
- No tracked changes.

**Interfaces:**
- Consumes: Phase A candidate SHA/tree and explicit Phase B/Phase D GitHub branch-push authorization.
- Produces: remote branch and open PR whose head equals the local candidate SHA.

- [ ] **Step 1: Stop for owner authorization**

Require a written receipt that names `codex/v1.0.0-release-closure`, the exact Phase A SHA, permits
initial and Phase D pushes to that branch plus PR creation/update, and explicitly forbids merge,
tag, and Release. Without that receipt, remain at `LOCAL_RELEASE_CANDIDATE_READY`.

- [ ] **Step 2: Revalidate fail-closed Git state**

Run:

```powershell
$Candidate = git rev-parse HEAD
if ((git branch --show-current) -ne "codex/v1.0.0-release-closure") { throw "Wrong branch" }
if (git status --porcelain) { throw "Dirty release worktree" }
git fetch origin --prune
if ((git rev-parse origin/main) -ne "8ce548b7b7ae2b812dbacadf477b6600e9d2d867") { throw "Unexpected main" }
git merge-base --is-ancestor origin/main $Candidate
if ($LASTEXITCODE -ne 0) { throw "NON_FAST_FORWARD_CANDIDATE" }
if (@(git rev-list --merges origin/main..$Candidate).Count -ne 0) { throw "NON_LINEAR_CANDIDATE" }
if (git ls-remote --heads origin codex/v1.0.0-release-closure) { throw "Remote branch already exists" }
```

Expected: exact base, clean branch, no merge commits, and no remote branch.

- [ ] **Step 3: Push the exact branch and create one PR**

Run only under the owner receipt:

```powershell
git push --set-upstream origin codex/v1.0.0-release-closure
$Body = @"
## Scope
- v1.0.0 source release candidate
- distinct authoritative adapter and merged model cards
- no research evidence, training, reward, protocol, evaluation, dataset, or weight changes

## Evidence boundary
- immutable evidence source: 1a077a45e309594e5bb43743a8b84d89155595d4
- protocol learning succeeded; strategy learning remained limited
- 13/463 wins (2.81%), not a practical Wordle solver

## Publication sequence
This PR remains open through the README-only HF transaction and immutable receipt closure.
"@
$PrUrl = gh pr create --repo kuotunyu/agentic-rl-wordle --base main --head codex/v1.0.0-release-closure --title "release: close v1.0.0 portfolio publication" --body $Body
```

Expected: one PR URL; no merge.

- [ ] **Step 4: Verify exact PR identity**

Run:

```powershell
$Pr = gh pr view --repo kuotunyu/agentic-rl-wordle --json number,state,mergeable,mergeStateStatus,baseRefOid,headRefOid,url | ConvertFrom-Json
if ($Pr.state -ne "OPEN") { throw "PR not open" }
if ($Pr.baseRefOid -ne "8ce548b7b7ae2b812dbacadf477b6600e9d2d867") { throw "PR base mismatch" }
if ($Pr.headRefOid -ne $Candidate) { throw "PR head mismatch" }
```

Expected: OPEN PR with exact base/head.

### Task 7: Require all six PR jobs to complete successfully

**Files:**
- No tracked changes.

**Interfaces:**
- Consumes: exact PR head and GitHub Actions run.
- Produces: verified job receipt and checkpoint `PR_OPEN_CI_GREEN_HF_UPDATE_REQUIRED`.

- [ ] **Step 1: Select the exact pull-request run**

Run:

```powershell
$Candidate = git rev-parse HEAD
$Runs = gh run list --repo kuotunyu/agentic-rl-wordle --branch codex/v1.0.0-release-closure --event pull_request --limit 20 --json databaseId,headSha,status,conclusion,url | ConvertFrom-Json
$Run = @($Runs | Where-Object { $_.headSha -eq $Candidate })
if ($Run.Count -ne 1) { throw "Expected exactly one PR run for exact head" }
gh run watch $Run[0].databaseId --repo kuotunyu/agentic-rl-wordle --exit-status --interval 5
```

Expected: watch exits zero.

- [ ] **Step 2: Fail closed on any non-success job**

Define `Assert-SixSuccessJobs` and `Test-SixSuccessValidatorFailsClosed` exactly as specified in
Shared GitHub verification interfaces, then run:

```powershell
Test-SixSuccessValidatorFailsClosed
$View = gh run view $Run[0].databaseId --repo kuotunyu/agentic-rl-wordle --json headSha,status,conclusion,jobs,url | ConvertFrom-Json
Assert-SixSuccessJobs -RunView $View -ExpectedSha $Candidate
```

Expected: the synthetic failure is rejected first; the live run then has the exact six-name set
and six `completed/success` conclusions.

- [ ] **Step 3: Confirm the PR remains open**

Run:

```powershell
$Pr = gh pr view --repo kuotunyu/agentic-rl-wordle --json state,headRefOid,url | ConvertFrom-Json
if ($Pr.state -ne "OPEN" -or $Pr.headRefOid -ne $Candidate) { throw "PR changed during CI" }
```

Expected: OPEN, exact head. Record `PR_OPEN_CI_GREEN_HF_UPDATE_REQUIRED` and stop for Phase C
owner authorization.

---

# Phase C — Hugging Face README-only transaction

Phase C requires a separate written owner authorization naming both repositories, both expected
parent revisions, the exact PR head, and README-only scope. Its terminal state is
`HF_README_ONLY_UPDATE_VERIFIED`.

### Task 8: Prove fail-closed HF validators and preflight both repositories

**Files:**
- No tracked changes.

**Interfaces:**
- Consumes: authenticated `huggingface_hub` client, exact Git card blobs, approved HF baselines.
- Produces: verified account identity, live full inventories, and pure validator self-test receipt.

- [ ] **Step 1: Stop for Phase C owner authorization**

Require written authorization for the exact PR head and these parent revisions:

- adapter `ef1e98ce214921049b86dce7c104c88875130023`;
- merged `a59a4fb4c26e5d0612ce3a3574193ec58d46fc64`.

The receipt must repeat that only `README.md` may change and that no weight may be downloaded or
uploaded. Without it, remain at `PR_OPEN_CI_GREEN_HF_UPDATE_REQUIRED`.

- [ ] **Step 2: Verify the authenticated library exists without installing anything**

Run:

```powershell
py -3.12 -c "import huggingface_hub; print(huggingface_hub.__version__)"
```

Expected: an installed version is printed. If import fails, stop for owner direction; do not add a
dependency or install an unreviewed package.

- [ ] **Step 3: Run pure fail-closed validator tests before network mutation**

Run this no-network, no-file-write command:

```powershell
@'
from copy import deepcopy

PARTIAL = "PARTIAL_HF_CARD_UPDATE"


def require_account(name: str) -> None:
    if name != "steven0226":
        raise RuntimeError("authenticated account mismatch")


def require_revision(actual: str, expected: str) -> None:
    if actual != expected:
        raise RuntimeError("baseline revision mismatch")


def require_post(pre: dict, post: dict, approved: bytes, remote: bytes) -> None:
    if set(pre) != set(post):
        raise RuntimeError("filename inventory changed")
    for path in pre:
        if path != "README.md" and pre[path] != post[path]:
            raise RuntimeError("non-README artifact changed")
    if approved != remote:
        raise RuntimeError("remote README bytes mismatch")


def state(adapter_ok: bool, merged_ok: bool) -> str:
    if adapter_ok and not merged_ok:
        return PARTIAL
    if adapter_ok and merged_ok:
        return "HF_README_ONLY_UPDATE_VERIFIED"
    return "HF_UNCHANGED"


def must_fail(callable_) -> None:
    try:
        callable_()
    except RuntimeError:
        return
    raise AssertionError("validator did not fail closed")


good = {
    "README.md": {"blob": "readme-old", "lfs": None, "size": 100},
    "model.safetensors": {"blob": "weight", "lfs": "sha256", "size": 300},
}
post = deepcopy(good)
post["README.md"] = {"blob": "readme-new", "lfs": None, "size": 120}
require_account("steven0226")
require_revision("expected", "expected")
require_post(good, post, b"approved", b"approved")
must_fail(lambda: require_account("different-account"))
must_fail(lambda: require_revision("changed", "expected"))
changed_names = deepcopy(post)
changed_names["extra.bin"] = {"blob": "extra", "lfs": None, "size": 1}
must_fail(lambda: require_post(good, changed_names, b"approved", b"approved"))
changed_blob = deepcopy(post)
changed_blob["model.safetensors"]["blob"] = "changed"
must_fail(lambda: require_post(good, changed_blob, b"approved", b"approved"))
changed_lfs = deepcopy(post)
changed_lfs["model.safetensors"]["lfs"] = "changed"
must_fail(lambda: require_post(good, changed_lfs, b"approved", b"approved"))
changed_size = deepcopy(post)
changed_size["model.safetensors"]["size"] = 301
must_fail(lambda: require_post(good, changed_size, b"approved", b"approved"))
must_fail(lambda: require_post(good, post, b"approved", b"different"))
assert state(True, False) == PARTIAL
assert state(True, True) == "HF_README_ONLY_UPDATE_VERIFIED"
assert state(False, False) == "HF_UNCHANGED"
print("fail-closed HF validator tests: PASS")
'@ | py -3.12 -
```

Expected: `fail-closed HF validator tests: PASS` and no network access.

- [ ] **Step 4: Run authenticated, read-only full inventory preflight**

Use a read-only Python command built from these exact approved identities:

```python
EXPECTED = {
    "steven0226/qwen2.5-1.5b-wordle-grpo": {
        "revision": "ef1e98ce214921049b86dce7c104c88875130023",
        "readme_sha256": "c3ab2ecbc0a032e77345239b02f41b967a8398017e312d7a2ea8e45a04afcf5b",
        "readme_size": 7246,
        "blobs": {
            ".gitattributes": "52373fe24473b1aa44333d318f578ae6bf04b49b",
            "adapter_config.json": "b1de78b261b03a020391d839400ff5664a009fd9",
            "adapter_model.safetensors": "8e4ffe7ef1ec47f9361fb94ad53d5fad338129b0",
            "chat_template.jinja": "bdf7919a96cfe43d50914a007b9c0877bd0ec27e",
            "README.md": "a2c7a02968a4566c344ebf92f8c45773fc7a8455",
            "tokenizer_config.json": "4d8760d91bde2ac751d25844835c33847a68cdf9",
            "tokenizer.json": "34510ff0037cd50428af467a17ead5a96140a32c",
            "training_args.bin": "4ab7d01ba69f79c9a8b301793796f9272e000a87",
        },
        "lfs": {
            "adapter_model.safetensors": ("92e6379ed7ddf363e7f500b143afa7a2dc725d3e86bd87bc9eb933831c7d68b7", 73911112),
            "tokenizer.json": ("3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8", 11421892),
            "training_args.bin": ("9df95d5c562cd327397015f3324e3627fd280725af4f10be014233835e778ab8", 7569),
        },
    },
    "steven0226/qwen2.5-1.5b-wordle-grpo-merged": {
        "revision": "a59a4fb4c26e5d0612ce3a3574193ec58d46fc64",
        "readme_sha256": "d3a35ef0db5324b3f67135e0cd216dbd980cd6afc0cf03aaf6621e36b9777e00",
        "readme_size": 7347,
        "blobs": {
            ".gitattributes": "52373fe24473b1aa44333d318f578ae6bf04b49b",
            "chat_template.jinja": "bdf7919a96cfe43d50914a007b9c0877bd0ec27e",
            "config.json": "97c2b63b467e3d0f1c22c493f19e81c2fd8b5318",
            "generation_config.json": "a8aca904d377977b666e4bd5d526356e627574bf",
            "model.safetensors": "d7d7779ec79579c35d69a7a0ca6ecdfec41c051a",
            "README.md": "581cb1e37f31b8d200c05576da0647eba12aa1ae",
            "tokenizer_config.json": "770e41d6c92519d525eede4cbcf3ba27f6425311",
            "tokenizer.json": "34510ff0037cd50428af467a17ead5a96140a32c",
        },
        "lfs": {
            "model.safetensors": ("b6c55086e798e1f62e6d970f07ee97ab39c1e0af3ee4b6ecdb2a349e485087af", 3087467144),
            "tokenizer.json": ("3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8", 11421892),
        },
    },
}
```

Run one Python stdin command containing the exact `EXPECTED` assignment above followed by this
body; it defines the snapshot interface reused by Task 9:

```python
import hashlib
import json
import urllib.parse
import urllib.request

from huggingface_hub import HfApi, get_token
from huggingface_hub.utils import build_hf_headers


def snapshot(api: HfApi, repo_id: str) -> dict[str, object]:
    info = api.model_info(repo_id, files_metadata=True)
    files: dict[str, dict[str, object]] = {}
    for sibling in info.siblings or []:
        lfs = sibling.lfs
        files[sibling.rfilename] = {
            "blob_id": sibling.blob_id,
            "size": sibling.size,
            "lfs_sha256": None if lfs is None else lfs.sha256,
            "lfs_size": None if lfs is None else lfs.size,
        }
    return {"repository": repo_id, "revision": info.sha, "files": files}


def readme_bytes(repo_id: str, revision: str) -> bytes:
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    url = f"https://huggingface.co/{quoted_repo}/resolve/{quoted_revision}/README.md"
    request = urllib.request.Request(url, headers=build_hf_headers(token=get_token()))
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def require_baseline(actual: dict[str, object], expected: dict[str, object]) -> None:
    if actual["revision"] != expected["revision"]:
        raise RuntimeError("baseline revision mismatch")
    files = actual["files"]
    if set(files) != set(expected["blobs"]):
        raise RuntimeError("baseline filename inventory mismatch")
    for path, blob_id in expected["blobs"].items():
        if files[path]["blob_id"] != blob_id:
            raise RuntimeError(f"baseline blob mismatch: {path}")
    for path, (sha256, size) in expected["lfs"].items():
        if files[path]["lfs_sha256"] != sha256 or files[path]["lfs_size"] != size:
            raise RuntimeError(f"baseline LFS mismatch: {path}")


api = HfApi()
account_name = (api.whoami() or {}).get("name")
if account_name != "steven0226":
    raise RuntimeError("authenticated account mismatch")
receipts: dict[str, dict[str, object]] = {}
for repo_id, expected in EXPECTED.items():
    actual = snapshot(api, repo_id)
    require_baseline(actual, expected)
    remote_readme = readme_bytes(repo_id, expected["revision"])
    if len(remote_readme) != expected["readme_size"]:
        raise RuntimeError(f"baseline README size mismatch: {repo_id}")
    if hashlib.sha256(remote_readme).hexdigest() != expected["readme_sha256"]:
        raise RuntimeError(f"baseline README content mismatch: {repo_id}")
    receipts[repo_id] = actual
print(json.dumps(receipts, indent=2, sort_keys=True))
print("HF_BASELINE_VERIFIED")
```

Concatenate the two adjacent Python blocks in this step without changing a byte and feed that
complete program to:

```powershell
py -3.12 -
if ($LASTEXITCODE -ne 0) { throw "HF_BASELINE_MISMATCH" }
```

The program calls no model-download API. It prints only public-safe normalized inventory JSON and
`HF_BASELINE_VERIFIED`; authorization headers and tokens remain in memory and are never printed.

Expected: both complete inventories match. Any mismatch stops before Task 9.

### Task 9: Execute adapter-first optimistic README transaction

**Files:**
- No tracked changes; one public-safe JSON receipt is written under the system temporary directory.

**Interfaces:**
- Consumes: exact PR head, authoritative Git card blob bytes, verified pre-inventories, existing HF credentials.
- Produces: adapter/merged post revisions, full pre/post inventory receipts, and state `HF_README_ONLY_UPDATE_VERIFIED` or a fail-closed stop state.

- [ ] **Step 1: Revalidate PR head and prepare a new receipt target**

Run:

```powershell
$Pr = gh pr view --repo kuotunyu/agentic-rl-wordle --json state,headRefOid | ConvertFrom-Json
$PrHead = git rev-parse HEAD
if ($Pr.state -ne "OPEN" -or $Pr.headRefOid -ne $PrHead) { throw "PR identity changed" }
$env:GIT_PR_HEAD = $PrHead
$env:HF_RECEIPT_FILE = Join-Path ([IO.Path]::GetTempPath()) "agentic-rl-wordle-hf-receipts.json"
if (Test-Path -LiteralPath $env:HF_RECEIPT_FILE) { throw "Receipt target already exists" }
```

- [ ] **Step 2: Run the single-operation transaction program**

Execute a Python stdin program that:

| Function signature | Exact contract |
|---|---|
| `git_blob(commit: str, relative: str) -> bytes` | Return `subprocess.check_output(["git", "cat-file", "blob", f"{commit}:{relative}"])`. |
| `snapshot(api: HfApi, repo_id: str) -> dict[str, object]` | Use the exact Task 8 body; return repository, live revision, and every filename/blob/size/LFS identity. |
| `require_baseline(actual: dict[str, object], expected: dict[str, object]) -> None` | Use the exact Task 8 body; raise on revision, filename, blob, LFS SHA, or LFS-size mismatch. |
| `require_post(pre: dict[str, object], post: dict[str, object], approved_readme: bytes, remote_readme: bytes) -> None` | Require identical filename sets and size-bearing inventory fields, identical non-README identities, and byte-equal README payload. |
| `commit_readme(api: HfApi, repo_id: str, parent_revision: str, message: str, payload: bytes) -> str` | Make exactly one `CommitOperationAdd` for `README.md`, pass `parent_commit`, and return `CommitInfo.oid`. |
| `write_public_receipt(path: Path, receipt: dict[str, object]) -> None` | Reject any key matching `token`, `credential`, `environment`, or `local_path`, then write UTF-8 JSON with sorted keys. |

No helper accepts or returns a token, local credential-store path, environment dump, or model
payload.

1. Defines the exact `EXPECTED` object from Task 8.
2. Reads card bytes with `git cat-file blob "$PrHead:docs/model_card.md"` and
   `git cat-file blob "$PrHead:docs/model_card_merged.md"` through `subprocess.check_output`.
3. Rechecks `HfApi().whoami()["name"] == "steven0226"`.
4. Rechecks both full baseline inventories immediately before mutation.
5. Calls `HfApi.create_commit` for the adapter with exactly:

```python
CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=io.BytesIO(adapter_card_bytes))
```

and:

```python
repo_id="steven0226/qwen2.5-1.5b-wordle-grpo"
repo_type="model"
commit_message="docs: update full-463 adapter model card"
parent_commit="ef1e98ce214921049b86dce7c104c88875130023"
```

6. Retrieves the adapter post revision, verifies the exact same filename set, exact equality of
   every non-README normalized identity, exact README bytes, and unchanged adapter/`training_args`
   LFS identities.
7. Only after step 6 succeeds, repeats the baseline check and one-operation commit for merged with:

```python
CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=io.BytesIO(merged_card_bytes))
```

and:

```python
repo_id="steven0226/qwen2.5-1.5b-wordle-grpo-merged"
repo_type="model"
commit_message="docs: update full-463 merged model card"
parent_commit="a59a4fb4c26e5d0612ce3a3574193ec58d46fc64"
```

8. Applies the same complete post verification to merged.
9. Writes public-safe JSON to `HF_RECEIPT_FILE` and validates it against this schema fragment:

```json
{
  "type": "object",
  "required": ["state", "git_pr_head", "adapter", "merged"],
  "properties": {
    "state": {"const": "HF_README_ONLY_UPDATE_VERIFIED"},
    "git_pr_head": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
    "adapter": {
      "type": "object",
      "required": ["pre", "post", "card_sha256"],
      "properties": {"card_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}
    },
    "merged": {
      "type": "object",
      "required": ["pre", "post", "card_sha256"],
      "properties": {"card_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}
    }
  }
}
```

The `pre` and `post` objects contain only repository, revision, filename, blob ID, size, LFS SHA,
and LFS size. The JSON contains no token, environment, local path, or credential metadata.

- [ ] **Step 3: Enforce partial-update behavior**

The transaction program handles outcomes exactly:

- adapter commit absent or failed: write state `HF_UNCHANGED`, stop, and do not call merged;
- adapter commit exists but verification fails: write `HF_ARTIFACT_INTEGRITY_FAILURE`, stop, and
  do not call merged;
- adapter verified and merged commit/verification fails: write `PARTIAL_HF_CARD_UPDATE` with the
  complete adapter receipt, perform no rollback, and stop before any Git push/merge/tag/Release;
- both verified: write `HF_README_ONLY_UPDATE_VERIFIED`.

No exception handler may attempt a second commit, force update, delete, or rollback.

- [ ] **Step 4: Validate the receipt and remote diff**

Run:

```powershell
$Receipt = Get-Content -LiteralPath $env:HF_RECEIPT_FILE -Raw | ConvertFrom-Json
if ($Receipt.state -ne "HF_README_ONLY_UPDATE_VERIFIED") { throw $Receipt.state }
if ($Receipt.git_pr_head -ne $PrHead) { throw "Receipt PR head mismatch" }
py -3.12 scripts/check_publication_boundary.py $env:HF_RECEIPT_FILE --identity-tip $PrHead
```

Expected: clean receipt, two changed revisions, README bytes equal card bytes, identical non-README
inventory, and unchanged all LFS identities. Record `HF_README_ONLY_UPDATE_VERIFIED`; do not push
or merge in this phase.

---

# Phase D — Git evidence closure

Phase D uses the Phase B authorization envelope for one update push to the same open PR. It does
not authorize merge. Its terminal state is `RELEASE_PR_READY_FOR_MERGE_REVIEW`.

### Task 10: Commit exact HF receipts into Git evidence

**Files:**
- Modify: `tests/test_release_closure.py`
- Modify: `docs/huggingface-audit.md`
- Modify: `docs/release-readiness.md`
- Modify: `CHANGELOG.md`
- Modify: `release/v1.0.0.md`

**Interfaces:**
- Consumes: `HF_RECEIPT_FILE` schema from Task 9.
- Produces: exact cross-document post-revision and card-hash evidence; authoritative cards remain byte-unchanged.

- [ ] **Step 1: Read and validate typed receipt fields**

Run:

```powershell
$Receipt = Get-Content -LiteralPath $env:HF_RECEIPT_FILE -Raw | ConvertFrom-Json
if ($Receipt.state -ne "HF_README_ONLY_UPDATE_VERIFIED") { throw "HF closure not verified" }
$AdapterPost = $Receipt.adapter.post.revision
$MergedPost = $Receipt.merged.post.revision
$AdapterCardHash = $Receipt.adapter.card_sha256
$MergedCardHash = $Receipt.merged.card_sha256
foreach ($Value in @($AdapterPost, $MergedPost)) { if ($Value -notmatch '^[0-9a-f]{40}$') { throw "Invalid HF revision" } }
foreach ($Value in @($AdapterCardHash, $MergedCardHash)) { if ($Value -notmatch '^[0-9a-f]{64}$') { throw "Invalid card hash" } }
```

- [ ] **Step 2: Add the failing post-receipt consistency test**

Append:

```python
def test_hf_closure_receipts_match_authoritative_cards_and_release_notes():
    audit = read_text("docs/huggingface-audit.md")
    readiness = read_text("docs/release-readiness.md")
    notes = read_text("release/v1.0.0.md")

    adapter_revision = extract_backticked_value("Post-update adapter revision", audit)
    merged_revision = extract_backticked_value("Post-update merged revision", audit)
    adapter_hash = extract_backticked_value("Post-update adapter README SHA-256", audit)
    merged_hash = extract_backticked_value("Post-update merged README SHA-256", audit)

    assert re.fullmatch(r"[0-9a-f]{40}", adapter_revision)
    assert re.fullmatch(r"[0-9a-f]{40}", merged_revision)
    assert adapter_revision != "ef1e98ce214921049b86dce7c104c88875130023"
    assert merged_revision != "a59a4fb4c26e5d0612ce3a3574193ec58d46fc64"
    assert adapter_hash == sha256_bytes(ADAPTER_CARD)
    assert merged_hash == sha256_bytes(MERGED_CARD)

    for literal in (adapter_revision, merged_revision, adapter_hash, merged_hash):
        assert literal in readiness
        assert literal in notes

    assert "HF_README_ONLY_UPDATE_VERIFIED" in readiness
    assert "PARTIAL_HF_CARD_UPDATE" in audit
```

Run:

```powershell
py -3.12 -m pytest -p no:cacheprovider tests/test_release_closure.py::test_hf_closure_receipts_match_authoritative_cards_and_release_notes -q
```

Expected: FAIL because the Git docs still contain only the pre-update state.

- [ ] **Step 3: Write literal receipt values with `apply_patch`**

Use `apply_patch` to replace the explicit pending state. Write four backticked literal values in
`docs/huggingface-audit.md` using this exact receipt-to-label mapping:

| Audit label | Receipt property |
|---|---|
| `Post-update adapter revision` | `$Receipt.adapter.post.revision` |
| `Post-update merged revision` | `$Receipt.merged.post.revision` |
| `Post-update adapter README SHA-256` | `$Receipt.adapter.card_sha256` |
| `Post-update merged README SHA-256` | `$Receipt.merged.card_sha256` |

The patch contains the validated literal property values, not PowerShell expressions. Add complete pre/post
inventory tables from the receipt and explicitly state that only README blob IDs changed. Copy
the same literal revision/hash values into readiness and release notes. Update the changelog with
two README-only HF commits. Do not modify either authoritative card.

- [ ] **Step 4: Run GREEN, remote-byte, and immutable-source gates**

Run:

```powershell
py -3.12 -m pytest -p no:cacheprovider tests/test_release_closure.py -q
py -3.12 scripts/check_publication_boundary.py --identity-tip HEAD
git diff --exit-code HEAD -- docs/model_card.md docs/model_card_merged.md
git diff --exit-code 1a077a45e309594e5bb43743a8b84d89155595d4 -- results baselines wordle_grpo_colab_train.ipynb wordle_full463_eval_colab.ipynb wordle_eval_push_colab.ipynb wordle_rl_bundle.sha256 src/wordle_rl/rewards.py src/wordle_rl/protocol.py eval
git diff --check
```

Also refetch each remote README at its post revision and assert byte equality with the corresponding
Git blob at the current head. Refetch both full inventories and compare every non-README normalized
identity with the receipt. Expected: all gates pass.

- [ ] **Step 5: Commit the immutable HF evidence closure**

Run:

```powershell
git add tests/test_release_closure.py docs/huggingface-audit.md docs/release-readiness.md CHANGELOG.md release/v1.0.0.md
git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit -m "docs: record immutable Hugging Face closure receipts"
```

Expected: one docs/tests commit; cards, source, workflow, and research artifacts are unchanged.

### Task 11: Update the open PR and re-require exact CI success

**Files:**
- No tracked changes.

**Interfaces:**
- Consumes: Phase D closure SHA and existing open PR.
- Produces: updated exact PR head with six successful jobs.

- [ ] **Step 1: Revalidate existing authorization and PR identity**

Require the Phase B receipt to cover this update push, then run:

```powershell
$ExpectedMain = "8ce548b7b7ae2b812dbacadf477b6600e9d2d867"
$ClosureHead = git rev-parse HEAD
$Pr = gh pr view codex/v1.0.0-release-closure --repo kuotunyu/agentic-rl-wordle --json number,state,baseRefOid,headRefOid,url | ConvertFrom-Json
if ($Pr.state -ne "OPEN" -or $Pr.baseRefOid -ne $ExpectedMain) { throw "PR base/state mismatch" }
git merge-base --is-ancestor $Pr.headRefOid $ClosureHead
if ($LASTEXITCODE -ne 0) { throw "Local closure is not a fast-forward of the current PR head" }
git fetch origin main
if ((git rev-parse origin/main) -ne $ExpectedMain) { throw "Origin main changed" }
if (git status --porcelain=v1) { throw "Release worktree is not clean" }
```

Expected: OPEN PR, unchanged base, old PR head is an ancestor of the clean local closure head.
Any mismatch stops before push.

- [ ] **Step 2: Push only the fast-forward branch update**

Run:

```powershell
$ClosureHead = git rev-parse HEAD
git push origin codex/v1.0.0-release-closure
```

Expected: ordinary fast-forward branch update; no force and no merge.

- [ ] **Step 3: Wait for and verify the exact new PR run**

Define the two shared CI functions exactly as specified above, then run:

```powershell
$Runs = gh run list --repo kuotunyu/agentic-rl-wordle --branch codex/v1.0.0-release-closure --event pull_request --limit 20 --json databaseId,headSha,status,conclusion,url | ConvertFrom-Json
$Run = @($Runs | Where-Object { $_.headSha -eq $ClosureHead })
if ($Run.Count -ne 1) { throw "Expected exactly one closure-head PR run" }
gh run watch $Run[0].databaseId --repo kuotunyu/agentic-rl-wordle --exit-status --interval 5
if ($LASTEXITCODE -ne 0) { throw "Closure-head PR run failed" }
Test-SixSuccessValidatorFailsClosed
$View = gh run view $Run[0].databaseId --repo kuotunyu/agentic-rl-wordle --json headSha,status,conclusion,jobs,url | ConvertFrom-Json
Assert-SixSuccessJobs -RunView $View -ExpectedSha $ClosureHead
$Pr = gh pr view $Pr.number --repo kuotunyu/agentic-rl-wordle --json state,headRefOid,url | ConvertFrom-Json
if ($Pr.state -ne "OPEN" -or $Pr.headRefOid -ne $ClosureHead) { throw "PR changed during closure CI" }
```

Expected terminal state: `RELEASE_PR_READY_FOR_MERGE_REVIEW`. Stop for Phase E owner authorization.

---

# Phase E — Stable closure

Phase E requires a new written owner authorization naming the exact release PR head and permitting
fast-forward integration, annotated tag creation, tag push, source-only GitHub Release, and scoped
cleanup. Its terminal state is `FROZEN / PORTFOLIO COMPLETE`.

### Task 12: Fast-forward protected main and require final main CI

**Files:**
- No tracked changes.

**Interfaces:**
- Consumes: exact Phase D PR head, green PR CI, protected main, Phase E authorization.
- Produces: local and remote `main` exactly equal to the release head; PR marked MERGED.

- [ ] **Step 1: Stop for exact merge/tag/Release authorization**

The receipt must name the exact closure head and explicitly permit only fast-forward integration,
annotated `v1.0.0`, tag CI, source-only Release, and final release-branch/worktree cleanup.

- [ ] **Step 2: Revalidate the merge candidate and branch protection**

Define all three Shared GitHub verification interfaces, set `$ClosureHead` to the exact SHA named
by the Phase E authorization, and run:

```powershell
git fetch origin main
$Base = git rev-parse origin/main
$Pr = gh pr view codex/v1.0.0-release-closure --repo kuotunyu/agentic-rl-wordle --json number,state,mergeable,mergeStateStatus,baseRefOid,headRefOid,url | ConvertFrom-Json
if ($Pr.state -ne "OPEN" -or $Pr.mergeable -ne "MERGEABLE" -or $Pr.mergeStateStatus -ne "CLEAN") { throw "PR is not OPEN/MERGEABLE/CLEAN" }
if ($Pr.baseRefOid -ne $Base -or $Pr.headRefOid -ne $ClosureHead) { throw "Authorized PR identity mismatch" }
git merge-base --is-ancestor $Base $ClosureHead
if ($LASTEXITCODE -ne 0) { throw "NON_FAST_FORWARD_CANDIDATE" }
$MergeCommits = @(git rev-list --min-parents=2 "$Base..$ClosureHead")
if ($MergeCommits.Count -ne 0) { throw "Candidate range contains a merge commit" }

$Runs = gh run list --repo kuotunyu/agentic-rl-wordle --branch codex/v1.0.0-release-closure --event pull_request --limit 20 --json databaseId,headSha | ConvertFrom-Json
$Run = @($Runs | Where-Object { $_.headSha -eq $ClosureHead })
if ($Run.Count -ne 1) { throw "Expected one exact-head PR run" }
Test-SixSuccessValidatorFailsClosed
$View = gh run view $Run[0].databaseId --repo kuotunyu/agentic-rl-wordle --json headSha,status,conclusion,jobs,url | ConvertFrom-Json
Assert-SixSuccessJobs -RunView $View -ExpectedSha $ClosureHead

$Protection = gh api repos/kuotunyu/agentic-rl-wordle/branches/main/protection | ConvertFrom-Json
Assert-MainProtection -Protection $Protection
```

Expected: every condition true. Otherwise stop `NON_FAST_FORWARD_CANDIDATE` or
`RELEASE_GATE_FAILED`.

- [ ] **Step 3: Advance main by fast-forward only**

Resolve the sole registered `main` worktree from Git metadata and run against that literal path:

```powershell
$WorktreeBlocks = ((git worktree list --porcelain) -join "`n") -split "(?:`r?`n){2,}"
$MainBlock = @($WorktreeBlocks | Where-Object { $_ -match '(?m)^branch refs/heads/main$' })
if ($MainBlock.Count -ne 1 -or $MainBlock[0] -notmatch '(?m)^worktree (.+)$') { throw "Expected one registered main worktree" }
$MainRepo = (Resolve-Path -LiteralPath $Matches[1]).Path
if ((git -C $MainRepo rev-parse --path-format=absolute --git-common-dir) -ne (git rev-parse --path-format=absolute --git-common-dir)) { throw "Main worktree belongs to a different repository" }
if ((git -C $MainRepo branch --show-current) -ne "main") { throw "Canonical worktree is not on main" }
if (git -C $MainRepo status --porcelain=v1) { throw "Canonical main worktree is not clean" }
git -C $MainRepo fetch origin --prune
git -C $MainRepo merge --ff-only $ClosureHead
if ((git -C $MainRepo rev-parse HEAD) -ne $ClosureHead) { throw "Local main mismatch" }
git -C $MainRepo push origin main
```

Expected: `main` advances to the exact authorized head without a merge commit; GitHub marks the PR
MERGED with that same SHA.

- [ ] **Step 4: Require exact final main push CI**

Define the two shared CI functions in the current shell, then run:

```powershell
$Runs = gh run list --repo kuotunyu/agentic-rl-wordle --branch main --event push --limit 20 --json databaseId,headBranch,headSha,status,conclusion,url | ConvertFrom-Json
$Run = @($Runs | Where-Object { $_.headBranch -eq "main" -and $_.headSha -eq $ClosureHead })
if ($Run.Count -ne 1) { throw "Expected exactly one final-main push run" }
gh run watch $Run[0].databaseId --repo kuotunyu/agentic-rl-wordle --exit-status --interval 5
if ($LASTEXITCODE -ne 0) { throw "Final-main CI failed" }
Test-SixSuccessValidatorFailsClosed
$View = gh run view $Run[0].databaseId --repo kuotunyu/agentic-rl-wordle --json headSha,status,conclusion,jobs,url | ConvertFrom-Json
Assert-SixSuccessJobs -RunView $View -ExpectedSha $ClosureHead
$Pr = gh pr view $Pr.number --repo kuotunyu/agentic-rl-wordle --json state,mergedAt,mergeCommit | ConvertFrom-Json
if ($Pr.state -ne "MERGED" -or $Pr.mergeCommit.oid -ne $ClosureHead) { throw "PR merge receipt mismatch" }
if ((git -C $MainRepo rev-parse HEAD) -ne $ClosureHead -or (git -C $MainRepo rev-parse origin/main) -ne $ClosureHead) { throw "Final main identity mismatch" }
```

Expected: six `completed/success` jobs; local `main == origin/main == $ClosureHead`.

### Task 13: Create and publish the annotated tag and source-only Release

**Files:**
- No tracked changes.

**Interfaces:**
- Consumes: exact final main SHA, green main CI, stable release notes.
- Produces: annotated tag object, green tag CI, and source-only GitHub Release with zero additional assets.

- [ ] **Step 1: Create the exact annotated tag locally**

Run:

```powershell
if (git tag --list v1.0.0) { throw "Tag already exists" }
git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" tag -a v1.0.0 $ClosureHead -m "agentic-rl-wordle v1.0.0"
$TagObject = git rev-parse v1.0.0
$Peeled = git rev-parse 'v1.0.0^{}'
if ($Peeled -ne $ClosureHead) { throw "Tag target mismatch" }
if ((git cat-file -t v1.0.0) -ne "tag") { throw "Tag is not annotated" }
```

Expected: annotated tag object distinct from and peeling to final main.

- [ ] **Step 2: Push only the tag and require tag CI**

Define the two shared CI functions, then run under the Phase E receipt:

```powershell
git push origin v1.0.0
$Runs = gh run list --repo kuotunyu/agentic-rl-wordle --branch v1.0.0 --event push --limit 20 --json databaseId,headBranch,headSha,status,conclusion,url | ConvertFrom-Json
$Run = @($Runs | Where-Object { $_.headBranch -eq "v1.0.0" -and $_.headSha -eq $ClosureHead })
if ($Run.Count -ne 1) { throw "Expected exactly one v1.0.0 push run" }
gh run watch $Run[0].databaseId --repo kuotunyu/agentic-rl-wordle --exit-status --interval 5
if ($LASTEXITCODE -ne 0) { throw "Tag CI failed" }
Test-SixSuccessValidatorFailsClosed
$View = gh run view $Run[0].databaseId --repo kuotunyu/agentic-rl-wordle --json headSha,status,conclusion,jobs,url | ConvertFrom-Json
Assert-SixSuccessJobs -RunView $View -ExpectedSha $ClosureHead
```

Expected: the tag run is the exact authorized commit and all six jobs are `completed/success`.

- [ ] **Step 3: Create the source-only GitHub Release**

Run:

```powershell
$ReleaseBodyPath = Join-Path ([IO.Path]::GetTempPath()) "agentic-rl-wordle-v1.0.0-release-body.md"
if (Test-Path -LiteralPath $ReleaseBodyPath) { throw "Release body target already exists" }
$CommittedBody = Get-Content -LiteralPath release/v1.0.0.md -Raw
$FinalIdentity = "Final main SHA: ``$ClosureHead```r`n`r`n"
[IO.File]::WriteAllText($ReleaseBodyPath, ($FinalIdentity + $CommittedBody), [Text.UTF8Encoding]::new($false))
py -3.12 scripts/check_publication_boundary.py $ReleaseBodyPath --identity-tip $ClosureHead
gh release create v1.0.0 --repo kuotunyu/agentic-rl-wordle --verify-tag --title "agentic-rl-wordle v1.0.0" --notes-file $ReleaseBodyPath
```

Do not pass any asset path. Then verify:

```powershell
$Release = gh release view v1.0.0 --repo kuotunyu/agentic-rl-wordle --json url,isDraft,isPrerelease,tagName,targetCommitish,assets,body | ConvertFrom-Json
if ($Release.isDraft -or $Release.isPrerelease) { throw "Release state mismatch" }
if ($Release.tagName -ne "v1.0.0") { throw "Release tag mismatch" }
if (@($Release.assets).Count -ne 0) { throw "Unexpected Release asset" }
if ($Release.body -notmatch [regex]::Escape($ClosureHead) -or
    $Release.body -notmatch '1a077a45e309594e5bb43743a8b84d89155595d4' -or
    $Release.body -notmatch '13/463' -or
    $Release.body -notmatch 'not a practical Wordle solver' -or
    $Release.body -notmatch 'aggregate-only' -or
    $Release.body -notmatch 'documentary lineage') { throw "Release body evidence mismatch" }
$ResolvedBody = (Resolve-Path -LiteralPath $ReleaseBodyPath).Path
$ResolvedTemp = (Resolve-Path -LiteralPath ([IO.Path]::GetTempPath())).Path
if ((Split-Path -Leaf $ResolvedBody) -ne "agentic-rl-wordle-v1.0.0-release-body.md" -or
    -not $ResolvedBody.StartsWith(($ResolvedTemp.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar), [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe release-body cleanup target" }
Remove-Item -LiteralPath $ResolvedBody
```

Expected: non-draft, non-prerelease Release, zero uploaded assets, an exact final-SHA receipt in
the public body, and no self-referential Git commit or retained temporary body.

### Task 14: Verify protection, identities, cleanup, and frozen state

**Files:**
- No tracked changes.

**Interfaces:**
- Consumes: completed GitHub/HF release receipts.
- Produces: clean canonical `main`, preserved tag/Release/HF cards, and no temporary release branch/worktree.

- [ ] **Step 1: Run the final immutable and external-state audit**

Define all Shared GitHub verification interfaces and run these fresh Git/GitHub reads:

```powershell
$WorktreeBlocks = ((git worktree list --porcelain) -join "`n") -split "(?:`r?`n){2,}"
$MainBlock = @($WorktreeBlocks | Where-Object { $_ -match '(?m)^branch refs/heads/main$' })
if ($MainBlock.Count -ne 1 -or $MainBlock[0] -notmatch '(?m)^worktree (.+)$') { throw "Expected one registered main worktree" }
$MainRepo = (Resolve-Path -LiteralPath $Matches[1]).Path
if ((git -C $MainRepo rev-parse --path-format=absolute --git-common-dir) -ne (git rev-parse --path-format=absolute --git-common-dir)) { throw "Main worktree belongs to a different repository" }
git -C $MainRepo fetch origin --prune --tags
$Final = git -C $MainRepo rev-parse HEAD
$Origin = git -C $MainRepo rev-parse origin/main
$Peeled = git -C $MainRepo rev-parse 'v1.0.0^{}'
if ($Final -ne $ClosureHead -or $Origin -ne $ClosureHead -or $Peeled -ne $ClosureHead) { throw "Final Git identity mismatch" }

foreach ($Branch in @("main", "v1.0.0")) {
  $Runs = gh run list --repo kuotunyu/agentic-rl-wordle --branch $Branch --event push --limit 20 --json databaseId,headBranch,headSha | ConvertFrom-Json
  $Run = @($Runs | Where-Object { $_.headBranch -eq $Branch -and $_.headSha -eq $ClosureHead })
  if ($Run.Count -ne 1) { throw "Final run count mismatch: $Branch" }
  $View = gh run view $Run[0].databaseId --repo kuotunyu/agentic-rl-wordle --json headSha,status,conclusion,jobs,url | ConvertFrom-Json
  Assert-SixSuccessJobs -RunView $View -ExpectedSha $ClosureHead
}

$Protection = gh api repos/kuotunyu/agentic-rl-wordle/branches/main/protection | ConvertFrom-Json
Assert-MainProtection -Protection $Protection
$OpenPrs = @(gh pr list --repo kuotunyu/agentic-rl-wordle --state open --limit 100 --json number | ConvertFrom-Json)
$Tags = @(gh api repos/kuotunyu/agentic-rl-wordle/git/matching-refs/tags/v1.0.0 | ConvertFrom-Json)
$Releases = @(gh release list --repo kuotunyu/agentic-rl-wordle --limit 100 --json tagName,isDraft,isPrerelease | ConvertFrom-Json)
if ($OpenPrs.Count -ne 0 -or $Tags.Count -ne 1 -or $Releases.Count -ne 1 -or $Releases[0].tagName -ne "v1.0.0") { throw "Final object-count mismatch" }

$Contributors = gh api --paginate --slurp repos/kuotunyu/agentic-rl-wordle/contributors | ConvertFrom-Json
$ContributorLogins = @($Contributors | ForEach-Object { $_ } | ForEach-Object { $_.login } | Sort-Object -Unique)
if ($ContributorLogins.Count -ne 1 -or $ContributorLogins[0] -ne "kuotunyu") { throw "Contributor identity mismatch" }
$IdentityLines = @(git -C $MainRepo log --format='%an <%ae>|%cn <%ce>' | Sort-Object -Unique)
if ($IdentityLines.Count -ne 1 -or $IdentityLines[0] -ne 'kuotunyu <61350295+kuotunyu@users.noreply.github.com>|kuotunyu <61350295+kuotunyu@users.noreply.github.com>') { throw "Commit identity mismatch" }
$Tagger = git -C $MainRepo for-each-ref refs/tags/v1.0.0 --format='%(taggername)|%(taggeremail)'
if ($Tagger -ne 'kuotunyu|<61350295+kuotunyu@users.noreply.github.com>') { throw "Tagger identity mismatch" }
```

Load the public-safe receipt, rerun the Task 8 `snapshot` and `readme_bytes` functions against
each receipt post revision, and require exact equality of every post filename/blob/size/LFS field.
Read the two approved card bytes from `$ClosureHead` with `git cat-file blob`; require those bytes
to equal the corresponding remote README bytes. The Python process exits nonzero on the first
field mismatch and prints only `HF_POST_STATE_VERIFIED` on success.

Finally rerun `test_research_artifacts_match_evidence_source`, the publication-boundary test, and
the Task 5 wheel/sdist metadata commands:

```powershell
Set-Location -LiteralPath $MainRepo
$env:PYTHONDONTWRITEBYTECODE = "1"
py -3.12 -m pytest -p no:cacheprovider tests/test_release_closure.py::test_research_artifacts_match_evidence_source tests/test_release_closure.py::test_release_docs_have_no_private_or_credential_leakage -q
$DistPath = [IO.Path]::GetFullPath((Join-Path $MainRepo "dist"))
if ((Split-Path -Leaf $DistPath) -ne "dist" -or -not $DistPath.StartsWith(([IO.Path]::GetFullPath($MainRepo) + [IO.Path]::DirectorySeparatorChar), [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe dist path" }
if (Test-Path -LiteralPath $DistPath) { Remove-Item -LiteralPath $DistPath -Recurse -Force }
py -3.12 -m build
py -3.12 -c "from pathlib import Path; import zipfile; w=list(Path('dist').glob('*.whl')); assert [p.name for p in w] == ['wordle_rl-1.0.0-py3-none-any.whl']; z=zipfile.ZipFile(w[0]); m=next(n for n in z.namelist() if n.endswith('.dist-info/METADATA')); assert 'Version: 1.0.0' in z.read(m).decode()"
py -3.12 -c "from pathlib import Path; import tarfile; s=list(Path('dist').glob('*.tar.gz')); assert [p.name for p in s] == ['wordle_rl-1.0.0.tar.gz']; t=tarfile.open(s[0]); m=next(x for x in t.getmembers() if x.name.endswith('/PKG-INFO')); assert b'Version: 1.0.0' in t.extractfile(m).read()"
```

Any mismatch stops cleanup so evidence remains inspectable.

- [ ] **Step 2: Delete only the merged remote branch**

Run:

```powershell
git push origin --delete codex/v1.0.0-release-closure
```

Expected: only the temporary remote release branch is removed.

- [ ] **Step 3: Remove the clean registered release worktree safely**

From canonical `main`, run:

```powershell
$ReleaseBlocks = ((git -C $MainRepo worktree list --porcelain) -join "`n") -split "(?:`r?`n){2,}"
$ReleaseBlock = @($ReleaseBlocks | Where-Object { $_ -match '(?m)^branch refs/heads/codex/v1\.0\.0-release-closure$' })
if ($ReleaseBlock.Count -ne 1 -or $ReleaseBlock[0] -notmatch '(?m)^worktree (.+)$') { throw "Expected one registered release worktree" }
$ReleaseWorktree = (Resolve-Path -LiteralPath $Matches[1]).Path
$ResolvedTemp = (Resolve-Path -LiteralPath ([IO.Path]::GetTempPath())).Path
if ((Split-Path -Leaf $ReleaseWorktree) -ne "agentic-rl-wordle-v1-release-closure-20260824" -or
    -not $ReleaseWorktree.StartsWith(($ResolvedTemp.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar), [StringComparison]::OrdinalIgnoreCase)) { throw "Unexpected release worktree path" }
if (git -C $ReleaseWorktree status --porcelain=v1) { throw "Release worktree is not clean" }
if ((git -C $ReleaseWorktree rev-parse HEAD) -ne $ClosureHead) { throw "Release worktree head mismatch" }
git -C $MainRepo worktree remove $ReleaseWorktree
```

Expected: Git unregisters and removes only the exact clean temporary worktree. Do not use a
recursive shell delete.

- [ ] **Step 4: Delete the merged local branch and temporary receipt**

Run:

```powershell
git -C $MainRepo branch -d codex/v1.0.0-release-closure
$ReceiptTarget = Join-Path ([IO.Path]::GetTempPath()) "agentic-rl-wordle-hf-receipts.json"
$ReceiptPath = (Resolve-Path -LiteralPath $ReceiptTarget).Path
$TempRoot = (Resolve-Path -LiteralPath ([IO.Path]::GetTempPath())).Path
if ((Split-Path -Leaf $ReceiptPath) -ne "agentic-rl-wordle-hf-receipts.json" -or
    -not $ReceiptPath.StartsWith(($TempRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar), [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe receipt cleanup target" }
Remove-Item -LiteralPath $ReceiptPath
```

Expected: branch deletion is non-forced; only the public-safe temporary receipt file is removed.

- [ ] **Step 5: Produce the final closure record**

Report final main SHA/tree, PR URL, main/tag run URLs and jobs, tag object/peeled commit, Release
URL, both HF pre/post revisions, unchanged weight identities, branch protection, contributor
identity, research-artifact equality, zero temporary branch/worktree, and clean canonical main.

Terminal state:

> `FROZEN / PORTFOLIO COMPLETE`

## Fail-closed acceptance matrix

| Required failure injection or live mismatch | Gate that must reject it | Required stop state |
|---|---|---|
| Git repository, worktree, branch, start HEAD, parent, or clean-state mismatch | Phase A preflight and Task 5 exact Git assertions | `BASELINE_MISMATCH` |
| HF baseline revision, filename inventory, blob, LFS SHA, or LFS size differs | Task 8 `require_revision`, `require_baseline`, and mutated inventory fixtures | `HF_BASELINE_MISMATCH` |
| `HfApi().whoami()["name"]` is not `steven0226` | Task 8 `require_account("different-account")` fixture and live preflight | `HF_BASELINE_MISMATCH` |
| Remote README bytes differ from the approved Git blob | Task 8 unequal-byte fixture and Task 9 `require_post` | `HF_CARD_IDENTITY_FAILURE` |
| Any non-README filename, blob, ordinary size, LFS SHA, or LFS size changes | Task 8 changed-name/blob/LFS/size fixtures and Task 9 full post snapshot | `HF_ARTIFACT_INTEGRITY_FAILURE` |
| Adapter succeeds but merged commit or verification fails | Task 8 `state(True, False)` assertion and Task 9 exception branch | `PARTIAL_HF_CARD_UPDATE` |
| PR, final-main, or tag workflow/job is not exact-head `completed/success` | `Test-SixSuccessValidatorFailsClosed` before every live `Assert-SixSuccessJobs` call | `RELEASE_GATE_FAILED`; tag failure is `STABLE_CLOSURE_FAILED` |
| Candidate is not a fast-forward descendant or contains a merge commit | Tasks 5, 6, and 12 `merge-base`/`rev-list` assertions | `NON_FAST_FORWARD_CANDIDATE` |
| A research artifact differs from evidence commit `1a077a45e309594e5bb43743a8b84d89155595d4` | `test_research_artifacts_match_evidence_source` plus Task 5 byte diff | `RESEARCH_EVIDENCE_BLOCKED` |
| Source, workflow, wheel, sdist, isolated import, or CLI distribution version differs | Stable-version test plus Tasks 5 and 14 metadata assertions | `RELEASE_GATE_FAILED` |
| Adapter/merged roles, metrics, bounded conclusion, or lineage wording are mixed or expanded | Distinct-card and bounded-claims release-contract tests | `RELEASE_GATE_FAILED` |
| A private path, token-shaped value, credential field, environment dump, or secret reaches cards, docs, artifacts, or receipts | publication-boundary tests, receipt-key rejection, artifact scan, and final scan | `RELEASE_GATE_FAILED` |

Every injected failure is exercised before its corresponding live parser or transaction is
trusted. None permits an automatic repair, evidence edit, HF rollback, merge, tag, or Release.

## Authorization checkpoints

1. **Implementation-entry authorization:** exact reviewed plan and then-current branch HEAD;
   permission to begin Phase A local implementation only; no external mutation.
2. **Phase B/Phase D GitHub branch/PR authorization:** exact Phase A SHA; initial push, PR creation,
   and one later closure-commit push only; no merge/tag/Release.
3. **Phase C HF authorization:** exact PR head and exact two parent revisions; two README-only
   commits only; no weight or non-README mutation.
4. **Phase E stable authorization:** exact Phase D SHA; fast-forward main, annotated tag, tag push,
   source-only Release, and scoped cleanup only.

No checkpoint inherits authority for a later checkpoint. At each terminal state the worker stops
and waits for the next written owner receipt.
