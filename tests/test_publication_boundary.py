import json
import subprocess

from scripts.check_publication_boundary import (
    scan_git_history,
    scan_git_identities,
    scan_text,
)


def _kinds(text: str, name: str = "fixture.txt") -> set[str]:
    return {finding.kind for finding in scan_text(name, text)}


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
