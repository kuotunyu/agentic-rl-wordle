"""TurnBasedGame 協定 + WordleGame adapter。

rollout.play_batch_episodes 只依賴這個小協定——Wordle 與煙霧環境（number_guess）
都實作它，訓練管線對兩者完全同構（階段 2 換環境 = 換 game factory 一行）。
"""

from __future__ import annotations

from typing import Protocol

from .env.wordle import WordleEnv
from .episode import EpisodeStats
from .knowledge import Knowledge
from .protocol import build_messages
from .runner import EpisodeState, play_turn


class TurnBasedGame(Protocol):
    done: bool

    def build_messages(self) -> list[dict]: ...
    def step(self, raw_text: str) -> bool: ...
    def stats(self) -> dict: ...
    def transcript(self) -> str: ...


class WordleGame:
    """把 runner 的 EpisodeState/play_turn 包成 TurnBasedGame。"""

    def __init__(
        self,
        answer: str,
        legal: frozenset[str],
        max_turns: int = 6,
        include_summary: bool = True,
    ):
        env = WordleEnv(legal=legal, max_turns=max_turns)
        env.reset(answer)
        self._state = EpisodeState(answer=answer, env=env, knowledge=Knowledge())
        self.max_turns = max_turns
        self.include_summary = include_summary

    @property
    def done(self) -> bool:
        return self._state.done

    def build_messages(self) -> list[dict]:
        return build_messages(
            self._state.history,
            max_turns=self.max_turns,
            include_summary=self.include_summary,
        )

    def step(self, raw_text: str) -> bool:
        play_turn(self._state, raw_text)
        return self._state.done

    def episode_stats(self) -> EpisodeStats:
        return EpisodeStats.from_turns(self._state.answer, self._state.history)

    def stats(self) -> dict:
        """episode 級純量，經 rollout_func 轉發給 reward functions。"""
        e = self.episode_stats()
        return {
            "win": float(e.won),
            "turns_used": e.turns_used,
            "num_illegal": e.num_illegal,
            "num_violation_turns": e.num_violation_turns,
            "num_repeats": e.num_repeats,
            "total_new_greens": e.total_new_greens,
            "total_new_presence": e.total_new_presence,
        }

    def transcript(self) -> str:
        return self.episode_stats().transcript_markdown()
