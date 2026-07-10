"""對局記錄結構：TurnRecord / EpisodeStats。

這是 baselines、eval、訓練 rollout、獎勵計算共用的「事實層」——
獎勵是 EpisodeStats 的純函數（rewards.py），評測指標是它的彙整（metrics.py）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .env.wordle import FEEDBACK_WIN, ILLEGAL


@dataclass(frozen=True)
class TurnRecord:
    turn: int                      # 1-based
    raw_output: str                # 模型（或 baseline agent）的原始輸出文字
    parse_outcome: str             # tag_ok | fallback_word | no_parse
    guess: str | None              # 正規化後的猜測（no_parse 時 None）
    feedback: str                  # "GYXXY" 或 ILLEGAL
    illegal: bool                  # 格式錯或非法詞（= feedback == ILLEGAL）
    illegal_reason: str | None     # none | bad_length | not_alpha | not_in_word_list
    violations: tuple[str, ...]    # 在 update 之前判定（reuses_absent_letter / breaks_green）
    had_info: bool                 # 本回合出手前，知識是否非空（違限率的分母）
    is_repeat: bool                # 與先前任一回合的 guess 完全相同
    new_greens: int                # 本回合新增綠位數（shaping 用）
    new_presence: int              # 本回合新增字母存在單位（shaping 用）


@dataclass
class EpisodeStats:
    answer: str
    won: bool
    turns_used: int
    num_illegal: int
    num_violation_turns: int       # 至少一項違規的回合數（獎勵 −1/回合 的口徑）
    num_repeats: int
    total_new_greens: int
    total_new_presence: int
    turns: list[TurnRecord] = field(default_factory=list)

    @classmethod
    def from_turns(cls, answer: str, turns: list[TurnRecord]) -> "EpisodeStats":
        won = any(t.feedback == FEEDBACK_WIN for t in turns)
        return cls(
            answer=answer,
            won=won,
            turns_used=len(turns),
            num_illegal=sum(1 for t in turns if t.illegal),
            num_violation_turns=sum(1 for t in turns if t.violations),
            num_repeats=sum(1 for t in turns if t.is_repeat),
            total_new_greens=sum(t.new_greens for t in turns),
            total_new_presence=sum(t.new_presence for t in turns),
            turns=list(turns),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def transcript_markdown(self) -> str:
        """單局人可讀 transcript（samples/ 與報告用）。"""
        lines = [f"### answer: `{self.answer}` — {'WIN' if self.won else 'LOSS'} in {self.turns_used} turn(s)", ""]
        for t in self.turns:
            fb = t.feedback if t.feedback != ILLEGAL else f"ILLEGAL ({t.illegal_reason})"
            flags = []
            if t.is_repeat:
                flags.append("repeat")
            flags.extend(t.violations)
            flag_str = f"  ⚠ {', '.join(flags)}" if flags else ""
            lines.append(f"- turn {t.turn}: guess=`{t.guess}` [{t.parse_outcome}] -> `{fb}`{flag_str}")
            raw = t.raw_output.strip().replace("\n", " ")
            if raw and raw != f"<guess>{t.guess}</guess>":
                lines.append(f"  - raw: {raw[:300]}")
        lines.append("")
        return "\n".join(lines)
