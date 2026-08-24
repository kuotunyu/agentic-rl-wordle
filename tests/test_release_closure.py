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


def git_blob_bytes(revision: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "cat-file", "blob", f"{revision}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout


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


def test_ci_jobs_running_full_pytest_fetch_immutable_evidence_history():
    workflow = read_text(".github/workflows/ci.yml")
    job_blocks = re.findall(
        r"^  ([a-z][a-z0-9-]*):\n(.*?)(?=^  [a-z][a-z0-9-]*:\n|\Z)",
        workflow,
        flags=re.MULTILINE | re.DOTALL,
    )
    pytest_jobs = {
        name: body for name, body in job_blocks if "      - run: python -m pytest\n" in body
    }

    assert set(pytest_jobs) == {"test-install"}
    for name, body in pytest_jobs.items():
        assert body.count(f"      - uses: actions/checkout@{CHECKOUT_SHA}\n") == 1, name
        assert (
            f"      - uses: actions/checkout@{CHECKOUT_SHA}\n"
            "        with:\n"
            "          fetch-depth: 0\n"
        ) in body, f"{name} must fetch the immutable evidence commit before pytest"


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


PRE_HF_LITERALS = (
    "ef1e98ce214921049b86dce7c104c88875130023",
    "a59a4fb4c26e5d0612ce3a3574193ec58d46fc64",
    "c3ab2ecbc0a032e77345239b02f41b967a8398017e312d7a2ea8e45a04afcf5b",
    "d3a35ef0db5324b3f67135e0cd216dbd980cd6afc0cf03aaf6621e36b9777e00",
    "92e6379ed7ddf363e7f500b143afa7a2dc725d3e86bd87bc9eb933831c7d68b7",
    "9df95d5c562cd327397015f3324e3627fd280725af4f10be014233835e778ab8",
    "b6c55086e798e1f62e6d970f07ee97ab39c1e0af3ee4b6ecdb2a349e485087af",
)


def test_hf_release_documents_preserve_baseline_and_record_verified_post_state():
    audit = read_text("docs/huggingface-audit.md")
    readiness = read_text("docs/release-readiness.md")
    changelog = read_text("CHANGELOG.md")
    notes = read_text("release/v1.0.0.md")

    for literal in PRE_HF_LITERALS:
        assert literal in audit, literal

    assert "HF README-only update pending owner authorization" not in audit
    assert "HF_README_ONLY_UPDATE_VERIFIED" in audit
    assert "HF_README_ONLY_UPDATE_VERIFIED" in readiness
    assert "1.0.0rc1" not in readiness
    assert "## [1.0.0] - 2026-08-24" in changelog
    assert "source-only" in notes
    assert "zero additional assets" in notes
    assert EVIDENCE_SHA in notes
    assert "exact upstream Qwen commit was not preserved" in audit
    assert "adapter-to-merged derivation is documentary lineage" in audit
    assert "Post-update adapter revision:" in audit
    assert "Post-update merged revision:" in audit


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
    assert adapter_hash == hashlib.sha256(git_blob_bytes("HEAD", "docs/model_card.md")).hexdigest()
    assert (
        merged_hash
        == hashlib.sha256(git_blob_bytes("HEAD", "docs/model_card_merged.md")).hexdigest()
    )

    for literal in (adapter_revision, merged_revision, adapter_hash, merged_hash):
        assert literal in readiness
        assert literal in notes

    assert "HF_README_ONLY_UPDATE_VERIFIED" in readiness
    assert "PARTIAL_HF_CARD_UPDATE" in audit


def test_model_card_receipt_hashes_ignore_crlf_worktree_representation():
    audit = read_text("docs/huggingface-audit.md")
    expected = {
        "docs/model_card.md": extract_backticked_value("Post-update adapter README SHA-256", audit),
        "docs/model_card_merged.md": extract_backticked_value(
            "Post-update merged README SHA-256", audit
        ),
    }

    for relative, receipt_hash in expected.items():
        committed = git_blob_bytes("HEAD", relative)
        crlf_worktree_representation = committed.replace(b"\n", b"\r\n")

        assert crlf_worktree_representation != committed
        assert hashlib.sha256(committed).hexdigest() == receipt_hash
        assert hashlib.sha256(crlf_worktree_representation).hexdigest() != receipt_hash
