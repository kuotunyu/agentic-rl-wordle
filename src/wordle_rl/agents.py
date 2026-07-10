"""三種 agent：Random / Heuristic / LLM。

統一介面：act_batch(states) -> 每個 state 一段「含 <guess>word</guess> 的原始文字」。
所有 agent 的輸出都走同一個 parser 與環境——指標對三者的定義完全一致
（baseline 的非法率理應 ~0，這本身就是管線的 sanity check）。
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .backends import GenConfig, GenerationBackend
    from .runner import EpisodeState


class Agent(Protocol):
    def act_batch(self, states: list["EpisodeState"]) -> list[str]: ...


class RandomAgent:
    """對整個合法猜測集（12,972）均勻抽樣——真·地板 baseline。"""

    def __init__(self, legal: frozenset[str], seed: int = 0):
        self._words = sorted(legal)
        self._rng = random.Random(seed)

    def act_batch(self, states: list["EpisodeState"]) -> list[str]:
        return [f"<guess>{self._rng.choice(self._words)}</guess>" for _ in states]


class HeuristicAgent:
    """字母頻率啟發式 + 線索過濾。

    候選 = 答案表（2,315）中與已知限制一致者；分數 = 逐位字母頻率和
    （頻率對「當前候選集」計算），單字內重複字母第二次起權重減半
    （避免早期挑 geese 這類低資訊詞）；猜 argmax、平手取字典序最小——全確定性。

    ⚠ 已文件化的 leakage：本 agent 看得到完整答案分布（含 eval 詞），
    是 Wordle solver 常規做法，作為「參照上界」使用，於 results/baselines.md 明載。
    """

    def __init__(self, answers: list[str]):
        self._answers = sorted(answers)

    def _pick(self, state: "EpisodeState") -> str:
        candidates = [w for w in self._answers if state.knowledge.is_consistent(w)]
        if not candidates:  # 只有環境/知識出 bug 才會發生；保守退回全答案表
            candidates = self._answers
        freq: list[dict[str, int]] = [{} for _ in range(5)]
        for w in candidates:
            for i, ch in enumerate(w):
                freq[i][ch] = freq[i].get(ch, 0) + 1

        def score(w: str) -> float:
            s = 0.0
            seen: set[str] = set()
            for i, ch in enumerate(w):
                weight = 1.0 if ch not in seen else 0.5
                s += weight * freq[i].get(ch, 0)
                seen.add(ch)
            return s

        return min(candidates, key=lambda w: (-score(w), w))

    def act_batch(self, states: list["EpisodeState"]) -> list[str]:
        return [f"<guess>{self._pick(s)}</guess>" for s in states]


class LLMAgent:
    """LLM 依協定遊玩：build_messages → backend 批次生成 → 原始文字交給 parser。"""

    def __init__(
        self,
        backend: "GenerationBackend",
        gen_config: "GenConfig",
        include_summary: bool = True,
        max_turns: int = 6,
    ):
        self.backend = backend
        self.gen_config = gen_config
        self.include_summary = include_summary
        self.max_turns = max_turns

    def act_batch(self, states: list["EpisodeState"]) -> list[str]:
        from .protocol import build_messages

        chats = [
            build_messages(
                s.history, max_turns=self.max_turns, include_summary=self.include_summary
            )
            for s in states
        ]
        return self.backend.generate(chats, self.gen_config)
