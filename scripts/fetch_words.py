"""抓取公開 Wordle 單字表到 data/（fetch-at-setup，不入 git）。

來源（2026-07-10 查證，位元組數/計數皆已驗證）：
- 答案 2,315 詞：cfreshman gist wordle-answers-alphabetical.txt
- 額外合法猜測 10,657 詞：cfreshman gist wordle-allowed-guesses.txt
  （合法猜測集 = 答案 ∪ 額外合法 = 12,972）
- 備援（僅供合法集側，MIT）：tabatkins/wordle-list 的 words 檔（14,855 = 現行 NYT 全猜測集）

注意：cfreshman 的 NYT 變體 gist（wordle-nyt-answers-alphabetical 等）已 404，勿用。
答案表沒有獨立備援來源——答案表抓不到就是硬錯誤（它定義了 train/eval 切分與獎勵）。

用法：
    python scripts/fetch_words.py            # 正常抓取 + 計數斷言
    python scripts/fetch_words.py --fallback # 合法集側改用 tabatkins（答案側不變）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ANSWERS_REVISION = "c46f451920d5cf6326d550fb2d6abb1642717852"
ALLOWED_REVISION = "d7c9e02d45afd26e12a71b4564189a949c29e8a9"
FALLBACK_REVISION = "255b9469c4dad99a3b95cc4ddbe139b3d3747868"

ANSWERS_URL = (
    "https://gist.githubusercontent.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b"
    f"/raw/{ANSWERS_REVISION}/wordle-answers-alphabetical.txt"
)
ALLOWED_URL = (
    "https://gist.githubusercontent.com/cfreshman/cdcdf777450c5b5301e439061d29694c"
    f"/raw/{ALLOWED_REVISION}/wordle-allowed-guesses.txt"
)
FALLBACK_LEGAL_URL = (
    f"https://raw.githubusercontent.com/tabatkins/wordle-list/{FALLBACK_REVISION}/words"
)

EXPECTED_ANSWERS = 2315
EXPECTED_ALLOWED = 10657
EXPECTED_LEGAL = 12972  # answers ∪ allowed
EXPECTED_FALLBACK_LEGAL = 14855
EXPECTED_ANSWERS_SHA256 = "5209b35f823f8b80f0404f863bd80df06d6a966c6eb1016d69f38badc6eed5d0"
EXPECTED_ALLOWED_SHA256 = "99be2e38dadf3e26952af7cb4d963f65b632d5de91aa99e5ce308e4dc9617b65"
EXPECTED_FALLBACK_SHA256 = "a7898cd20f36686d4c5b43ece226c36eb41a701df36da35bd18e21af41cfead4"

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "agentic-rl-wordle/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _normalize(raw: bytes) -> list[str]:
    words = []
    seen = set()
    for line in raw.decode("utf-8").splitlines():
        w = line.strip().lower()
        if len(w) == 5 and w.isalpha() and w.isascii() and w not in seen:
            seen.add(w)
            words.append(w)
    return sorted(words)


def _normalized_sha256(words: list[str]) -> str:
    return hashlib.sha256("\n".join(words).encode()).hexdigest()


def _matches_expected_hash(label: str, words: list[str], expected: str) -> bool:
    actual = _normalized_sha256(words)
    if actual == expected:
        return True
    print(
        f"FATAL: {label} normalized sha256 {actual} != pinned {expected}",
        file=sys.stderr,
    )
    return False


def _atomic_write(path: Path, words: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(words) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument(
        "--fallback",
        action="store_true",
        help="合法集側改抓 tabatkins/wordle-list（答案側仍必須是 cfreshman gist）",
    )
    args = ap.parse_args()
    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- 答案表：無備援，抓不到 = 硬錯誤 ---
    print(f"downloading answers  <- {ANSWERS_URL}")
    answers_raw = _download(ANSWERS_URL)
    answers = _normalize(answers_raw)
    if len(answers) != EXPECTED_ANSWERS:
        print(f"FATAL: answers count {len(answers)} != {EXPECTED_ANSWERS}", file=sys.stderr)
        return 1
    if not _matches_expected_hash("answers", answers, EXPECTED_ANSWERS_SHA256):
        return 1

    # --- 合法集側 ---
    if args.fallback:
        print(f"downloading legal(fallback) <- {FALLBACK_LEGAL_URL}")
        legal_raw = _download(FALLBACK_LEGAL_URL)
        legal_full = _normalize(legal_raw)
        if len(legal_full) != EXPECTED_FALLBACK_LEGAL:
            print(
                f"FATAL: fallback legal count {len(legal_full)} != {EXPECTED_FALLBACK_LEGAL}",
                file=sys.stderr,
            )
            return 1
        if not _matches_expected_hash("fallback legal", legal_full, EXPECTED_FALLBACK_SHA256):
            return 1
        answers_set = set(answers)
        allowed = sorted(w for w in legal_full if w not in answers_set)
        allowed_url = FALLBACK_LEGAL_URL
        allowed_revision = FALLBACK_REVISION
        source_kind = "fallback-tabatkins"
        source_content_sha256 = EXPECTED_FALLBACK_SHA256
    else:
        print(f"downloading allowed  <- {ALLOWED_URL}")
        allowed_raw = _download(ALLOWED_URL)
        allowed = _normalize(allowed_raw)
        if len(allowed) != EXPECTED_ALLOWED:
            print(f"FATAL: allowed count {len(allowed)} != {EXPECTED_ALLOWED}", file=sys.stderr)
            return 1
        if not _matches_expected_hash("allowed", allowed, EXPECTED_ALLOWED_SHA256):
            return 1
        allowed_url = ALLOWED_URL
        allowed_revision = ALLOWED_REVISION
        source_kind = "primary-cfreshman"
        source_content_sha256 = EXPECTED_ALLOWED_SHA256

    legal_n = len(set(answers) | set(allowed))
    if not args.fallback and legal_n != EXPECTED_LEGAL:
        print(f"FATAL: legal union {legal_n} != {EXPECTED_LEGAL}", file=sys.stderr)
        return 1

    _atomic_write(data_dir / "answers.txt", answers)
    _atomic_write(data_dir / "allowed.txt", allowed)

    source = {
        "kind": source_kind,
        "answers_url": ANSWERS_URL,
        "allowed_url": allowed_url,
        "answers_revision": ANSWERS_REVISION,
        "allowed_revision": allowed_revision,
        "source_content_sha256": source_content_sha256,
        "license_status": (
            "cfreshman-no-explicit-license; tabatkins-fallback-mit"
            if args.fallback
            else "cfreshman-no-explicit-license"
        ),
        "answers_count": len(answers),
        "allowed_count": len(allowed),
        "legal_count": legal_n,
        "answers_sha256": EXPECTED_ANSWERS_SHA256,
        "allowed_sha256": _normalized_sha256(allowed),
    }
    (data_dir / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")

    print(
        f"OK: answers={len(answers)} allowed={len(allowed)} legal={legal_n} "
        f"-> {data_dir} (SOURCE.json 已記錄出處與 sha256)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
