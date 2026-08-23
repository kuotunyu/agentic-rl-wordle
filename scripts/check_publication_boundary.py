"""Fail when public candidate files contain secrets or private machine residue."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

APPROVED_EMAILS = {"61350295+kuotunyu@users.noreply.github.com"}
APPROVED_GIT_IDENTITY = (
    "kuotunyu",
    "61350295+kuotunyu@users.noreply.github.com",
)
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".ipynb",
    ".jinja",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}
SECRET_PATTERNS = (
    ("hugging-face-token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("openai-token", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "private-key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
)
LOCAL_USER_PATH = re.compile(r"(?i)(?:[A-Z]:\\Users\\|file:///+[A-Z]:/Users/)[^\s\"']+")
EMAIL = re.compile(r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


@dataclass(frozen=True)
class Finding:
    location: str
    kind: str


def scan_text(name: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for kind, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            findings.append(Finding(name, kind))
    if LOCAL_USER_PATH.search(text):
        findings.append(Finding(name, "local-user-path"))
    for match in EMAIL.finditer(text):
        if match.group(0).lower() not in APPROVED_EMAILS:
            findings.append(Finding(name, "unapproved-email"))
            break

    if name.lower().endswith(".ipynb"):
        try:
            notebook = json.loads(text)
        except json.JSONDecodeError:
            findings.append(Finding(name, "invalid-notebook-json"))
        else:
            cells = notebook.get("cells", [])
            if any(cell.get("execution_count") is not None for cell in cells):
                findings.append(Finding(name, "notebook-execution-count"))
            if any(cell.get("outputs") for cell in cells):
                findings.append(Finding(name, "notebook-output"))

    return list(dict.fromkeys(findings))


def _scan_bytes(name: str, raw: bytes) -> list[Finding]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return []
    return scan_text(name, text)


def _scan_archive(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    if path.suffix in {".whl", ".zip"}:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                findings.extend(_scan_bytes(f"{path}!{member.filename}", archive.read(member)))
    elif path.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, mode="r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                stream = archive.extractfile(member)
                if stream is not None:
                    findings.extend(_scan_bytes(f"{path}!{member.name}", stream.read()))
    return findings


def scan_path(path: Path) -> list[Finding]:
    if path.is_dir():
        findings: list[Finding] = []
        for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            findings.extend(scan_path(child))
        return findings
    if path.suffix in {".whl", ".zip"} or path.name.endswith((".tar.gz", ".tgz")):
        return _scan_archive(path)
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE"}:
        return []
    return _scan_bytes(str(path), path.read_bytes())


def tracked_paths(repo: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    return [repo / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def scan_git_history(repo: Path) -> list[Finding]:
    rows = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    findings: list[Finding] = []
    seen: set[bytes] = set()
    for row in rows:
        object_id, _, raw_path = row.partition(b" ")
        if object_id in seen:
            continue
        seen.add(object_id)
        object_type = subprocess.run(
            ["git", "cat-file", "-t", object_id],
            cwd=repo,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if object_type != b"blob":
            continue
        path = raw_path.decode("utf-8", errors="replace") or "<unknown-path>"
        location = f"git:{object_id.decode()[:12]}:{path}"
        if Path(path).name.lower() == "interview.md":
            findings.append(Finding(location, "private-note-history"))
        raw = subprocess.run(
            ["git", "cat-file", "-p", object_id],
            cwd=repo,
            capture_output=True,
            check=True,
        ).stdout
        findings.extend(_scan_bytes(location, raw))
    return list(dict.fromkeys(findings))


def scan_git_identities(repo: Path) -> list[Finding]:
    rows = subprocess.run(
        ["git", "log", "--all", "--format=%H%x09%an%x09%ae%x09%cn%x09%ce"],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    ).stdout.splitlines()
    findings: list[Finding] = []
    for row in rows:
        commit, author_name, author_email, committer_name, committer_email = row.split("\t")
        author = (author_name, author_email.lower())
        committer = (committer_name, committer_email.lower())
        if author != APPROVED_GIT_IDENTITY or committer != APPROVED_GIT_IDENTITY:
            findings.append(Finding(f"git:commit:{commit[:12]}", "unapproved-git-identity"))
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="files/directories/archives to scan; defaults to Git tracked files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    paths = args.paths or tracked_paths(repo)
    findings: list[Finding] = []
    for path in paths:
        if path.name.lower() == "interview.md":
            findings.append(Finding(str(path), "private-note-tracked"))
        if path.exists():
            findings.extend(scan_path(path))
    if not args.paths:
        findings.extend(scan_git_history(repo))
        findings.extend(scan_git_identities(repo))

    findings = list(dict.fromkeys(findings))
    if findings:
        for finding in findings:
            print(f"PUBLICATION BOUNDARY FAIL: {finding.location}: {finding.kind}")
        return 1
    print(f"Publication boundary clean: scanned {len(paths)} top-level candidate path(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
