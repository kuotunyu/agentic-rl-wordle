"""Wordle 環境：score_guess 純函數 + WordleEnv。

回饋字元：G = 位置正確、Y = 存在但位置錯、X = 不存在（受重複字母計數限制）。
重複字母經典規則（兩趟）：先標 G 並從答案扣計數；再左→右對非綠位，
剩餘計數 > 0 標 Y 並遞減，否則 X。

非法猜測（None / 長度錯 / 非字母 / 不在合法集）**消耗回合**，回饋 = ILLEGAL 哨兵，
不洩漏任何字母資訊（也不更新線索——由呼叫端遵守）。
"""

from __future__ import annotations

from collections import Counter

FEEDBACK_WIN = "GGGGG"
ILLEGAL = "ILLEGAL"
WORD_LEN = 5
MAX_TURNS = 6


def score_guess(guess: str, answer: str) -> str:
    """經典兩趟重複字母計分。guess/answer 必須是 5 字母小寫英文。"""
    if len(guess) != WORD_LEN or not guess.isalpha() or guess != guess.lower():
        raise ValueError(f"guess 必須是 {WORD_LEN} 字母小寫英文：{guess!r}")
    if len(answer) != WORD_LEN or not answer.isalpha() or answer != answer.lower():
        raise ValueError(f"answer 必須是 {WORD_LEN} 字母小寫英文：{answer!r}")

    marks = [""] * WORD_LEN
    remaining: Counter[str] = Counter()
    for i in range(WORD_LEN):
        if guess[i] == answer[i]:
            marks[i] = "G"
        else:
            remaining[answer[i]] += 1
    for i in range(WORD_LEN):
        if marks[i]:
            continue
        if remaining[guess[i]] > 0:
            marks[i] = "Y"
            remaining[guess[i]] -= 1
        else:
            marks[i] = "X"
    return "".join(marks)


class WordleEnv:
    """reset(answer) → step(guess) → (feedback, done, info)。

    - 非法猜測消耗回合，feedback = ILLEGAL。
    - info["answer"] 只在 done 時提供（防洩漏）。
    - done 後再 step 拋 RuntimeError。
    """

    def __init__(self, legal: frozenset[str], max_turns: int = MAX_TURNS):
        if not legal:
            raise ValueError("legal 合法集不可為空")
        self.legal = legal
        self.max_turns = max_turns
        self._answer: str | None = None
        self.turn = 0
        self.done = False
        self.won = False
        self.history: list[tuple[str | None, str]] = []

    def reset(self, answer: str) -> dict:
        answer = answer.strip().lower()
        if len(answer) != WORD_LEN or not answer.isalpha() or not answer.isascii():
            raise ValueError(f"answer 必須是 {WORD_LEN} 字母英文：{answer!r}")
        self._answer = answer
        self.turn = 0
        self.done = False
        self.won = False
        self.history = []
        return {"turn": 0, "max_turns": self.max_turns, "history": []}

    def _classify(self, guess: str | None) -> tuple[str | None, str | None]:
        """回傳 (normalized_guess, illegal_reason)；合法時 illegal_reason=None。"""
        if guess is None:
            return None, "none"
        g = guess.strip().lower()
        if len(g) != WORD_LEN:
            return g, "bad_length"
        if not g.isalpha() or not g.isascii():
            return g, "not_alpha"
        if g not in self.legal:
            return g, "not_in_word_list"
        return g, None

    def step(self, guess: str | None) -> tuple[str, bool, dict]:
        if self._answer is None:
            raise RuntimeError("先 reset 再 step")
        if self.done:
            raise RuntimeError("episode 已結束，不可再 step")

        self.turn += 1
        norm, illegal_reason = self._classify(guess)
        if illegal_reason is not None:
            feedback = ILLEGAL
        else:
            assert norm is not None
            feedback = score_guess(norm, self._answer)
            if feedback == FEEDBACK_WIN:
                self.won = True

        self.done = self.won or self.turn >= self.max_turns
        self.history.append((norm, feedback))

        info = {
            "turn": self.turn,
            "turns_left": self.max_turns - self.turn,
            "guess": norm,
            "illegal": illegal_reason is not None,
            "illegal_reason": illegal_reason,
            "won": self.won,
        }
        if self.done:
            info["answer"] = self._answer
        return feedback, self.done, info
