"""score_guess 重複字母 14 案預驗證表 + WordleEnv 行為測試。

14 案在計畫階段已逐案人工用兩趟演算法追蹤核對（PLAN.md 階段 0 的表）。
"""

import pytest

from wordle_rl.env.wordle import FEEDBACK_WIN, ILLEGAL, WordleEnv, score_guess

# (answer, guess, expected, 驗證點)
DUPLICATE_CASES = [
    ("apple", "paper", "YYGYX", "交錯雙 p：G 吃掉一個，剩一個 Y"),
    ("abide", "speed", "XXYXY", "猜雙 e 答單 e：左先 Y、右 X（左到右封頂）"),
    ("those", "geese", "XXXGG", "G 先吃計數：唯一 e 被綠位吃掉，前面 e 全 X"),
    ("geese", "eagle", "YXYXG", "答三 e：綠吃一個，開頭 e 仍 Y"),
    ("abbey", "babes", "YYGGX", "雙 b 一對齊：G+Y"),
    ("abbey", "kebab", "XYGYY", "猜雙 b 答雙 b：G+尾 Y"),
    ("banal", "mamma", "XGXXY", "三 m 全 X；雙 a → 一 G 一 Y"),
    ("mamma", "mummy", "GXGGX", "答三 m 全綠化，其餘 X"),
    ("sleep", "eerie", "YYXXX", "猜三 e 答雙 e：恰兩 Y 第三 X"),
    ("hello", "llama", "YYXXX", "雙 l 對雙 l 無對齊：兩 Y 封頂"),
    ("hello", "label", "YXXYY", "雙 l 拆位：皆 Y"),
    ("dolly", "lolly", "XGGGG", "綠位吃光兩個 l → 開頭 l 是 X 不是 Y"),
    ("silly", "lolly", "XXGGG", "同上、兩個前導 X"),
    ("crane", "crane", "GGGGG", "全對即勝"),
]


@pytest.mark.parametrize("answer,guess,expected,why", DUPLICATE_CASES)
def test_score_guess_duplicate_table(answer, guess, expected, why):
    assert score_guess(guess, answer) == expected, why


def test_score_guess_rejects_bad_input():
    with pytest.raises(ValueError):
        score_guess("abcd", "crane")  # 長度錯
    with pytest.raises(ValueError):
        score_guess("crane", "abc")
    with pytest.raises(ValueError):
        score_guess("CRANE", "crane")  # 未小寫
    with pytest.raises(ValueError):
        score_guess("cr4ne", "crane")  # 非字母


LEGAL = frozenset({"crane", "slate", "paper", "apple", "hello", "label"})


def make_env(answer="crane", max_turns=6):
    env = WordleEnv(legal=LEGAL, max_turns=max_turns)
    env.reset(answer)
    return env


def test_win_flow():
    env = make_env("crane")
    feedback, done, info = env.step("slate")
    assert feedback == score_guess("slate", "crane")
    assert not done and not info["won"] and "answer" not in info
    feedback, done, info = env.step("crane")
    assert feedback == FEEDBACK_WIN
    assert done and info["won"] and info["answer"] == "crane"
    assert info["turn"] == 2


def test_illegal_word_consumes_turn_without_leaking():
    env = make_env("crane")
    feedback, done, info = env.step("zzzzz")  # 不在合法集
    assert feedback == ILLEGAL
    assert info["illegal"] and info["illegal_reason"] == "not_in_word_list"
    assert info["turn"] == 1 and not done


def test_none_guess_is_illegal():
    env = make_env()
    feedback, done, info = env.step(None)
    assert feedback == ILLEGAL and info["illegal_reason"] == "none"
    assert info["guess"] is None


def test_bad_length_and_not_alpha():
    env = make_env()
    _, _, info = env.step("abcdef")
    assert info["illegal_reason"] == "bad_length"
    _, _, info = env.step("ab1de")
    assert info["illegal_reason"] == "not_alpha"


def test_six_illegal_turns_is_a_loss_and_reveals_answer():
    env = make_env("crane")
    for i in range(6):
        feedback, done, info = env.step("zzzzz")
        assert feedback == ILLEGAL
    assert done and not info["won"]
    assert info["answer"] == "crane"
    assert info["turns_left"] == 0


def test_normalization_case_and_whitespace():
    env = make_env("crane")
    feedback, _, info = env.step("  CRANE  ")
    assert feedback == FEEDBACK_WIN and info["guess"] == "crane"


def test_step_after_done_raises():
    env = make_env("crane")
    env.step("crane")
    with pytest.raises(RuntimeError):
        env.step("slate")


def test_step_before_reset_raises():
    env = WordleEnv(legal=LEGAL)
    with pytest.raises(RuntimeError):
        env.step("crane")


def test_reset_validates_answer():
    env = WordleEnv(legal=LEGAL)
    with pytest.raises(ValueError):
        env.reset("abc")


def test_loss_after_max_turns():
    env = make_env("crane", max_turns=3)
    env.step("slate")
    env.step("paper")
    feedback, done, info = env.step("hello")
    assert done and not info["won"] and info["answer"] == "crane"
