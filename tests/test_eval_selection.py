from types import SimpleNamespace

import pytest

from eval.run_eval import select_eval_words


def _splits():
    return SimpleNamespace(
        eval_200=tuple(f"small-{i}" for i in range(200)),
        eval_full=tuple(f"full-{i}" for i in range(463)),
    )


def test_selects_full_held_out_split_without_200_word_truncation():
    words = select_eval_words(_splits(), "eval_full", 463)
    assert len(words) == 463
    assert words[-1] == "full-462"


def test_rejects_request_larger_than_selected_split():
    with pytest.raises(ValueError, match="eval_full"):
        select_eval_words(_splits(), "eval_200", 201)


def test_rejects_nonpositive_n():
    with pytest.raises(ValueError, match="positive"):
        select_eval_words(_splits(), "eval_full", 0)
