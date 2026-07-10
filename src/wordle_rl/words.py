"""單字表載入 + 固定 seed 切分。

- data/ 由 scripts/fetch_words.py 產生（answers.txt / allowed.txt / SOURCE.json）。
- 切分即時導出、不落地：train 1,852 / eval_full 463 / eval_200 = 洗牌尾段前 200。
  eval 詞永不參與訓練（紅線）。
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

EXPECTED_ANSWERS = 2315
EXPECTED_ALLOWED = 10657
EXPECTED_LEGAL = 12972

SPLIT_SEED = 42
TRAIN_FRACTION = 0.8
EVAL_SUBSET_SIZE = 200


class WordDataError(Exception):
    """單字表缺失或損毀。修復方式：python scripts/fetch_words.py"""


def _data_dir(data_dir: Path | None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    env_override = os.environ.get("WORDLE_RL_DATA_DIR")
    return Path(env_override) if env_override else DEFAULT_DATA_DIR


def _read_words(path: Path) -> list[str]:
    if not path.exists():
        raise WordDataError(f"{path} 不存在——先執行 python scripts/fetch_words.py")
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        w = line.strip().lower()
        if not w:
            continue
        if len(w) != 5 or not w.isalpha() or not w.isascii():
            raise WordDataError(f"{path} 含非法單字：{w!r}")
        words.append(w)
    if len(set(words)) != len(words):
        raise WordDataError(f"{path} 含重複單字")
    return words


def _source_kind(data_dir: Path) -> str:
    src = data_dir / "SOURCE.json"
    if src.exists():
        try:
            return json.loads(src.read_text(encoding="utf-8")).get("kind", "unknown")
        except json.JSONDecodeError:
            return "unknown"
    return "unknown"


def load_answers(data_dir: Path | None = None) -> list[str]:
    d = _data_dir(data_dir)
    answers = _read_words(d / "answers.txt")
    if len(answers) != EXPECTED_ANSWERS:
        raise WordDataError(
            f"answers.txt 有 {len(answers)} 詞，預期 {EXPECTED_ANSWERS}——來源可能變動，重跑 fetch_words.py"
        )
    return answers


def load_allowed(data_dir: Path | None = None) -> list[str]:
    d = _data_dir(data_dir)
    allowed = _read_words(d / "allowed.txt")
    # 主來源嚴格斷言；fallback（tabatkins）計數不同，SOURCE.json 記錄 kind 後放寬
    if _source_kind(d) == "primary-cfreshman" and len(allowed) != EXPECTED_ALLOWED:
        raise WordDataError(
            f"allowed.txt 有 {len(allowed)} 詞，預期 {EXPECTED_ALLOWED}——重跑 fetch_words.py"
        )
    if not allowed:
        raise WordDataError("allowed.txt 是空的")
    return allowed


def load_legal(data_dir: Path | None = None) -> frozenset[str]:
    d = _data_dir(data_dir)
    legal = frozenset(load_answers(d)) | frozenset(load_allowed(d))
    if _source_kind(d) == "primary-cfreshman" and len(legal) != EXPECTED_LEGAL:
        raise WordDataError(f"合法集聯集 {len(legal)} 詞，預期 {EXPECTED_LEGAL}")
    return legal


@dataclass(frozen=True)
class WordSplits:
    train: tuple[str, ...]
    eval_full: tuple[str, ...]
    eval_200: tuple[str, ...]


def get_splits(data_dir: Path | None = None, seed: int = SPLIT_SEED) -> WordSplits:
    """answers 依字母序 → Random(seed) 洗牌 → 前 80% train、後 20% eval。

    eval_200 = 洗牌尾段的前 200（無偏隨機子集，非字母序）。
    """
    answers = sorted(load_answers(data_dir))
    rng = random.Random(seed)
    rng.shuffle(answers)
    n_train = int(len(answers) * TRAIN_FRACTION)  # 2315 * 0.8 = 1852
    train = tuple(answers[:n_train])
    eval_full = tuple(answers[n_train:])
    return WordSplits(train=train, eval_full=eval_full, eval_200=eval_full[:EVAL_SUBSET_SIZE])
