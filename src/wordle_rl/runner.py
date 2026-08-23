"""推理側 lockstep 批次對局 runner——baselines 與 eval 共用（公平比較的關鍵）。

每回合對所有未結束的 episode 呼叫一次 agent.act_batch（LLM agent 因此
一回合只發一次批次生成；200 局最多 6 次批次呼叫跑完）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .agents import Agent
from .env.wordle import ILLEGAL, WordleEnv
from .episode import EpisodeStats, TurnRecord
from .knowledge import Knowledge, UpdateDelta
from .protocol import parse_guess


@dataclass
class EpisodeState:
    answer: str
    env: WordleEnv
    knowledge: Knowledge
    history: list[TurnRecord] = field(default_factory=list)
    prev_guesses: set[str] = field(default_factory=set)
    done: bool = False


def play_turn(state: EpisodeState, raw_text: str) -> TurnRecord:
    """把 agent 的一段原始輸出推進一個回合：parse → 違規判定 → env.step → 知識更新。

    順序不可換：violations 與 had_info 必須在 knowledge.update 之前取。
    """
    parsed = parse_guess(raw_text)
    had_info = state.knowledge.has_info()
    violations = tuple(sorted(state.knowledge.violations(parsed.guess))) if parsed.guess else ()

    feedback, done, info = state.env.step(parsed.guess)
    guess_norm = info["guess"]

    if guess_norm is not None and feedback != ILLEGAL:
        delta = state.knowledge.update(guess_norm, feedback)
    else:
        delta = UpdateDelta()

    is_repeat = guess_norm is not None and guess_norm in state.prev_guesses
    if guess_norm is not None:
        state.prev_guesses.add(guess_norm)

    record = TurnRecord(
        turn=info["turn"],
        raw_output=raw_text,
        parse_outcome=parsed.outcome,
        guess=guess_norm,
        feedback=feedback,
        illegal=info["illegal"],
        illegal_reason=info["illegal_reason"],
        violations=violations,
        had_info=had_info,
        is_repeat=is_repeat,
        new_greens=delta.new_greens,
        new_presence=delta.new_presence,
    )
    state.history.append(record)
    state.done = done
    return record


def run_episodes(
    agent: Agent,
    answers: list[str],
    legal: frozenset[str],
    max_turns: int = 6,
) -> list[EpisodeStats]:
    states = []
    for ans in answers:
        env = WordleEnv(legal=legal, max_turns=max_turns)
        env.reset(ans)
        states.append(EpisodeState(answer=ans, env=env, knowledge=Knowledge()))

    for _ in range(max_turns):
        active = [s for s in states if not s.done]
        if not active:
            break
        texts = agent.act_batch(active)
        if len(texts) != len(active):
            raise RuntimeError(f"agent 回傳 {len(texts)} 段文字，預期 {len(active)}")
        for state, text in zip(active, texts):
            play_turn(state, text)

    return [EpisodeStats.from_turns(s.answer, s.history) for s in states]
