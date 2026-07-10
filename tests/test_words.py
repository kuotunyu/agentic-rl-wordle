"""單字表載入與固定 seed 切分（需要 data/，由 scripts/fetch_words.py 產生）。"""

import pytest

from wordle_rl import words as W


@pytest.fixture(scope="module")
def splits():
    return W.get_splits()


def test_counts():
    assert len(W.load_answers()) == 2315
    assert len(W.load_allowed()) == 10657
    assert len(W.load_legal()) == 12972


def test_split_sizes(splits):
    assert len(splits.train) == 1852
    assert len(splits.eval_full) == 463
    assert len(splits.eval_200) == 200


def test_split_disjoint_and_complete(splits):
    train, ev = set(splits.train), set(splits.eval_full)
    assert not train & ev
    assert train | ev == set(W.load_answers())
    assert set(splits.eval_200) <= ev


def test_split_deterministic(splits):
    again = W.get_splits()
    assert again == splits


def test_eval_200_is_shuffled_prefix_not_alphabetical(splits):
    assert splits.eval_200 == splits.eval_full[:200]
    assert list(splits.eval_200) != sorted(splits.eval_200)  # 隨機子集，非字母序


def test_different_seed_differs():
    assert W.get_splits(seed=7).train != W.get_splits(seed=42).train


def test_all_answers_are_legal(splits):
    legal = W.load_legal()
    assert all(w in legal for w in splits.train[:50])
