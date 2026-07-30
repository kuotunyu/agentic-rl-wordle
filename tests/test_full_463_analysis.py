import pytest

from eval.analyze_full_463 import analyze, exact_mcnemar_base_zero


def _payload(base_wins=0, tuned_wins=13):
    return {
        "meta": {"n": 463, "seed": 42, "split": "eval_full"},
        "rows": {
            "model-base": {
                "n": 463,
                "wins": base_wins,
                "win_rate": base_wins / 463,
                "win_ci": [0.0, 0.0082],
                "illegal_rate": 1.0,
                "tag_ok_rate": 0.0,
            },
            "model +GRPO": {
                "n": 463,
                "wins": tuned_wins,
                "win_rate": tuned_wins / 463,
                "win_ci": [0.0165, 0.0474],
                "illegal_rate": 0.0018,
                "tag_ok_rate": 0.9985,
                "absent_reuse_rate": 0.585,
                "green_break_rate": 0.489,
                "repeat_rate": 0.00036,
            },
        },
    }


def test_exact_p_for_thirteen_one_direction_discordances():
    assert exact_mcnemar_base_zero(13) == pytest.approx(0.000244140625)


def test_analysis_applies_conservative_two_look_correction():
    result = analyze(_payload(), sequential_looks=2)
    assert result["n_pairs"] == 463
    assert result["tuned_wins"] == 13
    assert result["absolute_win_rate_gain"] == pytest.approx(13 / 463)
    assert result["bonferroni_corrected_p"] == pytest.approx(0.00048828125)
    assert result["tuned_legal_action_rate"] == pytest.approx(0.9982)


def test_aggregate_paired_analysis_rejects_nonzero_base_wins():
    with pytest.raises(ValueError, match="cannot recover"):
        analyze(_payload(base_wins=1))
