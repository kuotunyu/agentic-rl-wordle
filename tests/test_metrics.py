"""Wilson CI 與指標彙整。"""

import pytest

from wordle_rl.env.wordle import ILLEGAL
from wordle_rl.episode import EpisodeStats, TurnRecord
from wordle_rl.metrics import aggregate, wilson_ci


def test_wilson_ci_known_value():
    lo, hi = wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)


def test_wilson_ci_edge_cases():
    assert wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = wilson_ci(0, 200)
    assert lo == 0.0 and hi < 0.03
    lo, hi = wilson_ci(200, 200)
    assert hi == 1.0 and lo > 0.97


def mk_turn(turn, guess, feedback, **kw):
    d = dict(
        turn=turn,
        raw_output=f"<guess>{guess}</guess>",
        parse_outcome="tag_ok",
        guess=guess,
        feedback=feedback,
        illegal=feedback == ILLEGAL,
        illegal_reason=None,
        violations=(),
        had_info=turn > 1,
        is_repeat=False,
        new_greens=0,
        new_presence=0,
    )
    d.update(kw)
    return TurnRecord(**d)


def test_aggregate_basic():
    win_ep = EpisodeStats.from_turns(
        "crane",
        [mk_turn(1, "slate", "XXGXG"), mk_turn(2, "crane", "GGGGG")],
    )
    loss_ep = EpisodeStats.from_turns(
        "dolly",
        [
            mk_turn(1, "zzzzz", ILLEGAL, parse_outcome="fallback_word"),
            mk_turn(2, "speed", "XXXXY"),
            mk_turn(3, "speed", "XXXXY", is_repeat=True, violations=("reuses_absent_letter",)),
        ],
    )
    m = aggregate([win_ep, loss_ep])
    assert m.n_episodes == 2 and m.wins == 1
    assert m.win_rate == 0.5
    assert m.avg_guesses_on_wins == 2.0
    assert m.total_turns == 5
    assert m.illegal_rate == pytest.approx(1 / 5)
    assert m.tag_ok_rate == pytest.approx(4 / 5)
    # 有資訊回合 = turn>1 的 3 個；其中 1 個 reuses_absent_letter
    assert m.turns_with_info == 3
    assert m.absent_reuse_rate == pytest.approx(1 / 3)
    assert m.green_break_rate == 0.0
    assert m.repeat_rate == pytest.approx(1 / 5)


def test_aggregate_no_wins_avg_guesses_none():
    ep = EpisodeStats.from_turns("crane", [mk_turn(1, "slate", "XXGXG")])
    m = aggregate([ep])
    assert m.avg_guesses_on_wins is None
    assert m.as_row()["avg_guesses_on_wins"] == "—"


def test_episode_stats_from_turns():
    ep = EpisodeStats.from_turns(
        "crane",
        [
            mk_turn(1, "arise", "YGXXG", new_greens=2, new_presence=3),
            mk_turn(2, "crane", "GGGGG", new_greens=3, new_presence=2),
        ],
    )
    assert ep.won and ep.turns_used == 2
    assert ep.total_new_greens == 5 and ep.total_new_presence == 5
    assert ep.num_illegal == 0 and ep.num_repeats == 0
