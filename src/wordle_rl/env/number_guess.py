"""階段 2 煙霧環境：猜 1~100 的數字，回饋 HIGHER / LOWER，≤7 回合。

目的：用最小環境驗證整條多輪 GRPO 管線（協定 → lockstep rollout → 獎勵 → 更新）。
二分搜尋 7 步必勝 → 未訓練模型有充分上升空間，A100 上 ~20 分鐘就該看到學習訊號。
與 Wordle 共用 <guess>N</guess> 協定形狀與 rollout 機制，只換環境與獎勵。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

LO, HI = 1, 100
MAX_TURNS = 7

NUMBER_SYSTEM_PROMPT = """You are playing a number guessing game. Find the secret whole number between 1 and 100 (inclusive) within 7 guesses.

After each guess you receive feedback:
- HIGHER: the secret number is higher than your guess.
- LOWER: the secret number is lower than your guess.

Rules:
- Every guess must be a whole number between 1 and 100.
- A badly formatted reply wastes a turn.
- Think briefly if needed (at most one short sentence), then output exactly one guess formatted as <guess>N</guess>."""

_TURN_INSTRUCTION = "Turn {turn} of {max_turns}. Output your guess as <guess>N</guess>."

_NUM_TAG_RE = re.compile(r"<guess>\s*(\d{1,3})\s*</guess>", re.IGNORECASE)
_TAG_MARKUP_RE = re.compile(r"</?\s*guess\s*>", re.IGNORECASE)
_NUM_RE = re.compile(r"\b(\d{1,3})\b")


def parse_number_guess(text: str) -> tuple[int | None, str]:
    """回傳 (guess, outcome)；outcome ∈ tag_ok / fallback_word / no_parse。"""
    tags = _NUM_TAG_RE.findall(text)
    if tags:
        n = int(tags[-1])
        if LO <= n <= HI:
            return n, "tag_ok"
    cleaned = _TAG_MARKUP_RE.sub(" ", text)
    nums = [int(m) for m in _NUM_RE.findall(cleaned) if LO <= int(m) <= HI]
    if nums:
        return nums[-1], "fallback_word"
    return None, "no_parse"


@dataclass
class NumberGuessGame:
    """自足的煙霧遊戲：實作 rollout.TurnBasedGame 協定。"""

    answer: int
    max_turns: int = MAX_TURNS
    turns: list[dict] = field(default_factory=list)
    done: bool = False
    won: bool = False

    def build_messages(self) -> list[dict]:
        messages = [
            {"role": "system", "content": NUMBER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Guess the secret number. "
                + _TURN_INSTRUCTION.format(turn=1, max_turns=self.max_turns),
            },
        ]
        for i, t in enumerate(self.turns):
            canonical = f"<guess>{t['guess']}</guess>" if t["guess"] is not None else t["raw"]
            messages.append({"role": "assistant", "content": canonical})
            if t["guess"] is None:
                fb = "Your reply did not contain a valid <guess>N</guess> tag. This turn is wasted."
            elif t["feedback"] == "HIGHER":
                fb = f"The secret number is HIGHER than {t['guess']}."
            elif t["feedback"] == "LOWER":
                fb = f"The secret number is LOWER than {t['guess']}."
            else:
                fb = "Correct!"
            messages.append(
                {
                    "role": "user",
                    "content": fb
                    + "\n"
                    + _TURN_INSTRUCTION.format(turn=i + 2, max_turns=self.max_turns),
                }
            )
        return messages

    def step(self, raw_text: str) -> bool:
        if self.done:
            raise RuntimeError("episode 已結束")
        guess, outcome = parse_number_guess(raw_text)
        if guess is None:
            feedback = "ILLEGAL"
        elif guess == self.answer:
            feedback = "CORRECT"
            self.won = True
        elif guess < self.answer:
            feedback = "HIGHER"
        else:
            feedback = "LOWER"
        prev = {t["guess"] for t in self.turns if t["guess"] is not None}
        self.turns.append(
            {
                "raw": raw_text,
                "guess": guess,
                "outcome": outcome,
                "feedback": feedback,
                "is_repeat": guess is not None and guess in prev,
            }
        )
        self.done = self.won or len(self.turns) >= self.max_turns
        return self.done

    def stats(self) -> dict:
        return {
            "win": float(self.won),
            "turns_used": len(self.turns),
            "num_illegal": sum(1 for t in self.turns if t["guess"] is None),
            "num_repeats": sum(1 for t in self.turns if t["is_repeat"]),
        }

    def transcript(self) -> str:
        lines = [
            f"### answer: {self.answer} — {'WIN' if self.won else 'LOSS'} in {len(self.turns)} turn(s)"
        ]
        for i, t in enumerate(self.turns, 1):
            lines.append(f"- turn {i}: guess={t['guess']} [{t['outcome']}] -> {t['feedback']}")
        return "\n".join(lines)


def number_guess_reward(stats: dict, max_turns: int = MAX_TURNS) -> float:
    """煙霧環境獎勵：勝 +5 + 0.5×剩餘回合；格式錯 −2/回合；重複 −2/次。

    回饋本身已密集（每回合都有方向資訊），不需要額外資訊 shaping。
    """
    r = 0.0
    if stats["win"]:
        r += 5.0 + 0.5 * (max_turns - stats["turns_used"])
    r += -2.0 * stats["num_illegal"]
    r += -2.0 * stats["num_repeats"]
    return round(r, 6)
