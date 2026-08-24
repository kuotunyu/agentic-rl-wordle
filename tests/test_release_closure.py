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
        worktree_blob = subprocess.run(
            ["git", "hash-object", f"--path={relative}", relative],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        evidence_blob = subprocess.run(
            ["git", "rev-parse", f"{EVIDENCE_SHA}:{relative}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert worktree_blob == evidence_blob, relative
