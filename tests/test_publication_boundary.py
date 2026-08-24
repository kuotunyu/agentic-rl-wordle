import json
import os
import subprocess
from pathlib import Path

from scripts.check_publication_boundary import (
    Finding,
    scan_git_history,
    scan_git_identities,
    scan_text,
)


def _kinds(text: str, name: str = "fixture.txt") -> set[str]:
    return {finding.kind for finding in scan_text(name, text)}


APPROVED_NAME = "kuotunyu"
APPROVED_EMAIL = "61350295+kuotunyu@users.noreply.github.com"
GITHUB_EMAIL = "noreply" + "@github.com"
OTHER_EMAIL = "other" + "@example.com"


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout.strip()


def _identity_tip_fixture(repo: Path) -> dict[str, str]:
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", APPROVED_NAME)
    _git(repo, "config", "user.email", APPROVED_EMAIL)

    tracked = repo / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "approved base")
    base = _git(repo, "rev-parse", "HEAD")

    tracked.write_text("branch head\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "approved branch head")
    approved_head = _git(repo, "rev-parse", "HEAD")
    branch_tree = _git(repo, "rev-parse", f"{approved_head}^{{tree}}")

    synthetic_env = os.environ.copy()
    synthetic_env.update(
        {
            "GIT_AUTHOR_NAME": APPROVED_NAME,
            "GIT_AUTHOR_EMAIL": APPROVED_EMAIL,
            "GIT_COMMITTER_NAME": "GitHub",
            "GIT_COMMITTER_EMAIL": GITHUB_EMAIL,
        }
    )
    synthetic = subprocess.run(
        ["git", "commit-tree", branch_tree, "-p", base, "-p", approved_head],
        cwd=repo,
        env=synthetic_env,
        input="synthetic pull request merge\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout.strip()
    _git(repo, "update-ref", "refs/pull/1/merge", synthetic)

    tracked.write_text("unapproved branch commit\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(
        repo,
        "commit",
        "-qm",
        "unapproved branch commit",
        "--author",
        f"Other <{OTHER_EMAIL}>",
    )
    unapproved_head = _git(repo, "rev-parse", "HEAD")
    blob = _git(repo, "rev-parse", f"{approved_head}:tracked.txt")

    # Local default scans must trust the checked-out publication tip, not every ref.
    _git(repo, "checkout", "--detach", "-q", approved_head)
    return {
        "approved_head": approved_head,
        "synthetic": synthetic,
        "unapproved_head": unapproved_head,
        "blob": blob,
    }


def test_scanner_rejects_secret_private_path_email_and_private_key():
    fake_hf_token = "hf_" + "A" * 32
    local_path = "C:" + "\\Users\\personal-name\\project"
    private_key = "-----BEGIN " + "PRIVATE KEY-----"
    unapproved_email = "person" + "@example.com"

    assert "hugging-face-token" in _kinds(fake_hf_token)
    assert "local-user-path" in _kinds(local_path)
    assert "private-key" in _kinds(private_key)
    assert "unapproved-email" in _kinds(unapproved_email)


def test_scanner_allows_placeholders_generic_drive_path_and_formal_identity():
    safe = "\n".join(
        [
            'token = userdata.get("HF_TOKEN")',
            "/content/drive/MyDrive/agentic-rl-wordle",
            "61350295+kuotunyu@users.noreply.github.com",
            "https://huggingface.co/steven0226/qwen2.5-1.5b-wordle-grpo",
        ]
    )

    assert _kinds(safe) == set()


def test_scanner_rejects_saved_notebook_outputs():
    notebook = json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "outputs": [{"output_type": "stream", "text": ["private output"]}],
                    "source": ["print('hello')"],
                }
            ]
        }
    )

    assert _kinds(notebook, "run.ipynb") == {
        "notebook-execution-count",
        "notebook-output",
    }


def test_history_scan_finds_secret_removed_from_worktree(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "kuotunyu"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "config",
            "user.email",
            "61350295+kuotunyu@users.noreply.github.com",
        ],
        cwd=tmp_path,
        check=True,
    )
    secret_file = tmp_path / "removed.txt"
    secret_file.write_text("hf_" + "A" * 32, encoding="utf-8")
    subprocess.run(["git", "add", "removed.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "add fixture"], cwd=tmp_path, check=True)
    secret_file.unlink()
    subprocess.run(["git", "add", "-u"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "remove fixture"], cwd=tmp_path, check=True)

    findings = scan_git_history(tmp_path)

    assert {finding.kind for finding in findings} == {"hugging-face-token"}


def test_git_identity_scan_rejects_unapproved_author(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "kuotunyu"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "config",
            "user.email",
            "61350295+kuotunyu@users.noreply.github.com",
        ],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "file.txt").write_text("safe", encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True)
    author = "Other <" + "other" + "@example.com>"
    subprocess.run(
        ["git", "commit", "-qm", "fixture", "--author", author],
        cwd=tmp_path,
        check=True,
    )

    findings = scan_git_identities(tmp_path)

    assert {finding.kind for finding in findings} == {"unapproved-git-identity"}


def test_default_identity_scan_uses_head_ancestry_not_all_refs(tmp_path):
    fixture = _identity_tip_fixture(tmp_path)

    findings = scan_git_identities(tmp_path)

    assert findings == [], (
        "HEAD is the approved branch tip, but git log --all also scanned the synthetic merge "
        f"{fixture['synthetic'][:12]} and an unrelated unapproved ref"
    )


def test_explicit_branch_tip_excludes_synthetic_merge_ref(tmp_path):
    fixture = _identity_tip_fixture(tmp_path)

    findings = scan_git_identities(tmp_path, fixture["approved_head"])

    assert findings == []


def test_explicit_synthetic_merge_tip_still_rejects_github_committer(tmp_path):
    fixture = _identity_tip_fixture(tmp_path)

    findings = scan_git_identities(tmp_path, fixture["synthetic"])

    assert findings == [
        Finding(
            f"git:commit:{fixture['synthetic'][:12]}",
            "unapproved-git-identity",
        )
    ]


def test_branch_ancestry_still_rejects_real_unapproved_commit(tmp_path):
    fixture = _identity_tip_fixture(tmp_path)

    findings = scan_git_identities(tmp_path, fixture["unapproved_head"])

    assert {finding.kind for finding in findings} == {"unapproved-git-identity"}
    assert findings[0].location == f"git:commit:{fixture['unapproved_head'][:12]}"


def test_identity_tip_fails_closed_for_empty_missing_and_noncommit(tmp_path):
    fixture = _identity_tip_fixture(tmp_path)

    for invalid_tip in ("", "missing-publication-tip", fixture["blob"]):
        findings = scan_git_identities(tmp_path, invalid_tip)
        assert {finding.kind for finding in findings} == {"invalid-git-identity-tip"}


def test_malformed_identity_log_row_fails_closed(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", APPROVED_NAME)
    _git(tmp_path, "config", "user.email", APPROVED_EMAIL)
    malformed_env = os.environ.copy()
    malformed_env.update(
        {
            "GIT_AUTHOR_NAME": "Malformed\tAuthor",
            "GIT_AUTHOR_EMAIL": APPROVED_EMAIL,
            "GIT_COMMITTER_NAME": APPROVED_NAME,
            "GIT_COMMITTER_EMAIL": APPROVED_EMAIL,
        }
    )
    _git(tmp_path, "commit", "--allow-empty", "-qm", "malformed identity row", env=malformed_env)

    findings = scan_git_identities(tmp_path)

    assert {finding.kind for finding in findings} == {"invalid-git-identity-log"}


def test_identity_scan_requires_exact_email_case(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", APPROVED_NAME)
    _git(tmp_path, "config", "user.email", APPROVED_EMAIL.upper())
    _git(tmp_path, "commit", "--allow-empty", "-qm", "wrong identity case")

    findings = scan_git_identities(tmp_path)

    assert {finding.kind for finding in findings} == {"unapproved-git-identity"}


def test_quality_workflow_tests_merge_result_but_scans_event_publication_tip():
    workflow = (Path(__file__).parents[1] / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    quality = workflow.split("  quality:\n", 1)[1].split("\n  test-install:\n", 1)[0]

    assert "fetch-depth: 0" in quality
    assert "ref:" not in quality
    assert (
        "PUBLICATION_IDENTITY_TIP: ${{ github.event_name == 'pull_request' "
        "&& github.event.pull_request.head.sha || github.sha }}"
    ) in quality
    assert (
        'python scripts/check_publication_boundary.py --identity-tip "$PUBLICATION_IDENTITY_TIP"'
    ) in quality
    assert "continue-on-error" not in quality
