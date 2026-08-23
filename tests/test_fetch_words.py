import hashlib
import json
import sys

from scripts import fetch_words as F


def _normalized_hash(words: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(words)).encode()).hexdigest()


def _configure_small_fixture(monkeypatch):
    answers = ["cigar", "rebut"]
    allowed = ["sissy"]
    monkeypatch.setattr(F, "EXPECTED_ANSWERS", 2)
    monkeypatch.setattr(F, "EXPECTED_ALLOWED", 1)
    monkeypatch.setattr(F, "EXPECTED_LEGAL", 3)
    monkeypatch.setattr(F, "EXPECTED_ANSWERS_SHA256", _normalized_hash(answers), raising=False)
    monkeypatch.setattr(F, "EXPECTED_ALLOWED_SHA256", _normalized_hash(allowed), raising=False)
    return answers, allowed


def test_fetch_rejects_same_count_but_modified_answers(monkeypatch, tmp_path):
    _, allowed = _configure_small_fixture(monkeypatch)
    altered_answers = ["cigar", "humph"]
    downloads = iter(
        [
            ("\n".join(altered_answers) + "\n").encode(),
            ("\n".join(allowed) + "\n").encode(),
        ]
    )
    monkeypatch.setattr(F, "_download", lambda _url: next(downloads))
    monkeypatch.setattr(sys, "argv", ["fetch_words.py", "--data-dir", str(tmp_path)])

    assert F.main() == 1


def test_source_metadata_records_pinned_revisions_and_hashes(monkeypatch, tmp_path):
    answers, allowed = _configure_small_fixture(monkeypatch)
    downloads = iter(
        [
            ("\n".join(answers) + "\n").encode(),
            ("\n".join(allowed) + "\n").encode(),
        ]
    )
    monkeypatch.setattr(F, "_download", lambda _url: next(downloads))
    monkeypatch.setattr(sys, "argv", ["fetch_words.py", "--data-dir", str(tmp_path)])

    assert F.main() == 0
    source = json.loads((tmp_path / "SOURCE.json").read_text(encoding="utf-8"))
    assert source["answers_revision"] == F.ANSWERS_REVISION
    assert source["allowed_revision"] == F.ALLOWED_REVISION
    assert source["answers_sha256"] == F.EXPECTED_ANSWERS_SHA256
    assert source["allowed_sha256"] == F.EXPECTED_ALLOWED_SHA256
