"""線索追蹤器：從 (guess, feedback) 累積已知限制。

計數語義（重複字母陷阱的正確處理）：
- 對單一 guess 中的字母 L，令 g/y/x = 該字母在此 guess 得到的 G/Y/X 數。
- min_counts[L] = max(舊值, g+y)：答案至少含 g+y 個 L。
- **x > 0 ⇒ exact_counts[L] = g+y**：出現 X 表示 L 的數量已探底——
  素 X（g+y=0）代表「不存在」；重複字母的第二份 X 代表「恰好 g+y 個」。
  所以「重用已判 X 的字母」的違規判定只能看 exact_counts == 0，
  絕不能看「這字母曾拿過 X」（那會把 dolly 猜 lolly 的開頭 X 誤判成 l 不存在）。
- 位置：G 記 greens[i]；任何非 G 的出現位記 not_at[i]（Y 與 X 都證明 L 不在 i）。

ILLEGAL 回合不揭露任何資訊——呼叫端不應對其呼叫 update()（update 亦自我防衛）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .env.wordle import ILLEGAL, WORD_LEN


@dataclass(frozen=True)
class UpdateDelta:
    """單次 update 的新資訊量（供獎勵 shaping，只算首次發現）。

    - new_greens：新增的綠位數（含 Y→G 升級——位置資訊是新的）。
    - new_presence：sum(min_counts) 的增量 = 新發現的「答案含此字母（再多一份）」單位。
      直接猜中綠位的新字母同時 +1 green 與 +1 presence（位置與存在都是新資訊）。
    上界：new_greens 總和 ≤ 5、new_presence 總和 ≤ 5。
    """

    new_greens: int = 0
    new_presence: int = 0


@dataclass
class Knowledge:
    greens: dict[int, str] = field(default_factory=dict)
    min_counts: dict[str, int] = field(default_factory=dict)
    exact_counts: dict[str, int] = field(default_factory=dict)
    not_at: dict[int, set[str]] = field(default_factory=dict)

    def has_info(self) -> bool:
        return bool(self.greens or self.min_counts or self.exact_counts or self.not_at)

    def update(self, guess: str, feedback: str) -> UpdateDelta:
        if feedback == ILLEGAL:
            return UpdateDelta()
        if len(guess) != WORD_LEN or len(feedback) != WORD_LEN:
            raise ValueError(f"guess/feedback 長度必須為 {WORD_LEN}：{guess!r}/{feedback!r}")

        prev_green = len(self.greens)
        prev_presence = sum(self.min_counts.values())

        # 逐字母統計本次 G/Y/X
        marks_by_letter: dict[str, list[str]] = {}
        for i in range(WORD_LEN):
            marks_by_letter.setdefault(guess[i], []).append(feedback[i])

        for letter, marks in marks_by_letter.items():
            g = marks.count("G")
            y = marks.count("Y")
            x = marks.count("X")
            self.min_counts[letter] = max(self.min_counts.get(letter, 0), g + y)
            if x > 0:
                self.exact_counts[letter] = g + y
            if self.min_counts[letter] == 0:
                del self.min_counts[letter]  # 不留 0 值雜訊

        for i in range(WORD_LEN):
            if feedback[i] == "G":
                self.greens[i] = guess[i]
            else:
                self.not_at.setdefault(i, set()).add(guess[i])

        return UpdateDelta(
            new_greens=len(self.greens) - prev_green,
            new_presence=sum(self.min_counts.values()) - prev_presence,
        )

    def violations(self, guess: str) -> set[str]:
        """對「即將猜出的 guess」判定違規（須在 update 之前呼叫）。

        - reuses_absent_letter：guess 含已知數量恰為 0 的字母。
        - breaks_green：已知綠位沒沿用。
        """
        out: set[str] = set()
        if any(self.exact_counts.get(c) == 0 for c in set(guess)):
            out.add("reuses_absent_letter")
        if any(guess[i] != letter for i, letter in self.greens.items()):
            out.add("breaks_green")
        return out

    def is_consistent(self, word: str) -> bool:
        """word 是否與所有已知限制相容（供啟發式 baseline 過濾候選）。"""
        for i, letter in self.greens.items():
            if word[i] != letter:
                return False
        for letter, n in self.min_counts.items():
            if word.count(letter) < n:
                return False
        for letter, n in self.exact_counts.items():
            if word.count(letter) != n:
                return False
        for i, excluded in self.not_at.items():
            if word[i] in excluded:
                return False
        return True
