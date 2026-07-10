"""獎勵純函數：量級、界限、防 hacking 論證的可執行斷言。"""

import pytest

from wordle_rl.episode import EpisodeStats
from wordle_rl.rewards import (
    DEFAULT_CONFIG,
    MAX_SHAPING,
    binary_reward,
    episode_reward,
    shaped_reward,
)


def ep(**kw) -> EpisodeStats:
    d = dict(
        answer="crane",
        won=False,
        turns_used=6,
        num_illegal=0,
        num_violation_turns=0,
        num_repeats=0,
        total_new_greens=0,
        total_new_presence=0,
        turns=[],
    )
    d.update(kw)
    return EpisodeStats(**d)


def test_perfect_first_turn_win():
    # 首猜全中：勝利 10 + 剩 5 回合加成 + 5 新綠 + 5 新存在
    s = ep(won=True, turns_used=1, total_new_greens=5, total_new_presence=5)
    assert shaped_reward(s) == pytest.approx(10 + 5 + 1.0 + 0.5)


def test_turn6_win_no_bonus():
    s = ep(won=True, turns_used=6, total_new_greens=5, total_new_presence=5)
    assert shaped_reward(s) == pytest.approx(10 + 0 + 1.5)


def test_all_garbage_loss():
    s = ep(num_illegal=6)
    assert shaped_reward(s) == pytest.approx(-12.0)


def test_repeat_spam_penalty():
    s = ep(num_repeats=3, total_new_presence=2)
    assert shaped_reward(s) == pytest.approx(-6.0 + 0.2)


def test_violation_penalty():
    s = ep(num_violation_turns=2, total_new_greens=1, total_new_presence=1)
    assert shaped_reward(s) == pytest.approx(-2.0 + 0.2 + 0.1)


def test_shaping_bound_and_dominance():
    """防 hacking 核心論證：刷滿 shaping 不求勝（1.5）<< 最差獲勝（10）。"""
    assert MAX_SHAPING == pytest.approx(1.5)
    max_farm = shaped_reward(ep(total_new_greens=4, total_new_presence=5))  # 不可能 5 綠不勝
    worst_win = shaped_reward(ep(won=True, turns_used=6))
    assert worst_win - max_farm >= 8.5
    # 拖延無利可圖：同 shaping 下，早勝嚴格優於晚勝
    early = shaped_reward(ep(won=True, turns_used=2, total_new_greens=5, total_new_presence=5))
    late = shaped_reward(ep(won=True, turns_used=5, total_new_greens=5, total_new_presence=5))
    assert early > late


def test_single_turn_penalty_exceeds_single_turn_shaping():
    """單回合非法（−2）必須壓過單回合可得的最大 shaping（5G+5存在不可能一回合全新——
    首猜全綠 = 0.2×5+0.1×5=1.5 但那也是獲勝回合；非勝回合單回合上界 0.8+0.5 < 2。"""
    assert abs(DEFAULT_CONFIG.illegal_penalty) >= 1.3


def test_binary_preset():
    assert binary_reward(ep(won=True, turns_used=6)) == 1.0
    assert binary_reward(ep(won=False, num_illegal=6)) == 0.0
    assert episode_reward(ep(won=True, turns_used=3), "binary") == 1.0


def test_episode_reward_dispatch_and_unknown():
    s = ep(won=True, turns_used=1, total_new_greens=5, total_new_presence=5)
    assert episode_reward(s, "shaped") == shaped_reward(s)
    with pytest.raises(ValueError):
        episode_reward(s, "nope")
