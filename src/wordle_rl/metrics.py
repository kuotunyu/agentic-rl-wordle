"""評測指標彙整：勝率（Wilson 95% CI）、勝局均猜、非法率、違限率。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .episode import EpisodeStats


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval。n=0 回 (0.0, 1.0)。"""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


@dataclass(frozen=True)
class AggregateMetrics:
    n_episodes: int
    wins: int
    win_rate: float
    win_ci_low: float
    win_ci_high: float
    avg_guesses_on_wins: float | None   # 勝局平均消耗回合（含浪費回合）；無勝局為 None
    total_turns: int
    illegal_rate: float                 # 規格定義的 union：非 tag_ok「或」環境判非法 / 全部回合
    env_illegal_rate: float             # 只算環境判非法（feedback == ILLEGAL）——與獎勵 −2 的口徑一致
    tag_ok_rate: float                  # parse_outcome == tag_ok 的回合占比（格式面，單獨分列）
    turns_with_info: int
    absent_reuse_rate: float | None     # 重用已知不存在字母 / 有資訊回合；分母 0 為 None
    green_break_rate: float | None      # 沒沿用已知綠位 / 有資訊回合
    repeat_rate: float                  # 重複猜測回合 / 全部回合

    def as_row(self) -> dict:
        def pct(x: float | None) -> str:
            return "—" if x is None else f"{100 * x:.1f}%"

        return {
            "win_rate": f"{100 * self.win_rate:.1f}% [{100 * self.win_ci_low:.1f}, {100 * self.win_ci_high:.1f}]",
            "avg_guesses_on_wins": "—" if self.avg_guesses_on_wins is None else f"{self.avg_guesses_on_wins:.2f}",
            "illegal_rate": pct(self.illegal_rate),
            "tag_ok_rate": pct(self.tag_ok_rate),
            "absent_reuse_rate": pct(self.absent_reuse_rate),
            "green_break_rate": pct(self.green_break_rate),
            "repeat_rate": pct(self.repeat_rate),
        }


def aggregate(episodes: list[EpisodeStats]) -> AggregateMetrics:
    n = len(episodes)
    wins = sum(1 for e in episodes if e.won)
    ci_low, ci_high = wilson_ci(wins, n)

    all_turns = [t for e in episodes for t in e.turns]
    total_turns = len(all_turns)
    info_turns = [t for t in all_turns if t.had_info]

    win_turn_counts = [e.turns_used for e in episodes if e.won]

    def _rate(count: int, denom: int) -> float:
        return count / denom if denom else 0.0

    return AggregateMetrics(
        n_episodes=n,
        wins=wins,
        win_rate=_rate(wins, n),
        win_ci_low=ci_low,
        win_ci_high=ci_high,
        avg_guesses_on_wins=(sum(win_turn_counts) / len(win_turn_counts)) if win_turn_counts else None,
        total_turns=total_turns,
        illegal_rate=_rate(
            sum(1 for t in all_turns if t.illegal or t.parse_outcome != "tag_ok"), total_turns
        ),
        env_illegal_rate=_rate(sum(1 for t in all_turns if t.illegal), total_turns),
        tag_ok_rate=_rate(sum(1 for t in all_turns if t.parse_outcome == "tag_ok"), total_turns),
        turns_with_info=len(info_turns),
        absent_reuse_rate=(
            _rate(sum(1 for t in info_turns if "reuses_absent_letter" in t.violations), len(info_turns))
            if info_turns
            else None
        ),
        green_break_rate=(
            _rate(sum(1 for t in info_turns if "breaks_green" in t.violations), len(info_turns))
            if info_turns
            else None
        ),
        repeat_rate=_rate(sum(1 for t in all_turns if t.is_repeat), total_turns),
    )
