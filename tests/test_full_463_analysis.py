import json
from pathlib import Path

import pytest

from eval.analyze_full_463 import analyze, exact_mcnemar_base_zero
from wordle_rl.metrics import wilson_ci


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
                "illegal_rate": 5 / 2753,
                "tag_ok_rate": 2749 / 2753,
                "absent_reuse_rate": 1340 / 2290,
                "green_break_rate": 1119 / 2290,
                "repeat_rate": 1 / 2753,
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
    assert result["tuned_legal_action_rate"] == pytest.approx(2748 / 2753)


def test_analysis_recomputes_wilson_interval_instead_of_trusting_input():
    payload = _payload()
    payload["rows"]["model +GRPO"]["win_ci"] = [0.0, 1.0]

    result = analyze(payload)

    assert result["base_win_rate_wilson_ci_95"] == pytest.approx(wilson_ci(0, 463))
    assert result["tuned_win_rate_wilson_ci_95"] == pytest.approx(wilson_ci(13, 463))


def test_committed_aggregate_recovers_action_counts_and_statistics():
    report = Path(__file__).parents[1] / "results" / "full_463_report.json"
    result = analyze(json.loads(report.read_text(encoding="utf-8")))

    assert result["base_wins"] == 0
    assert result["tuned_wins"] == 13
    assert result["tuned_action_counts"] == {
        "total_turns": 2753,
        "protocol_adherent": 2749,
        "legal": 2748,
        "repeats": 1,
        "turns_with_info": 2290,
        "absent_reuses": 1340,
        "green_breaks": 1119,
    }
    assert result["mcnemar_two_sided_exact_p"] == pytest.approx(0.000244140625)
    assert result["bonferroni_corrected_p"] == pytest.approx(0.00048828125)


def test_aggregate_paired_analysis_rejects_nonzero_base_wins():
    with pytest.raises(ValueError, match="cannot recover"):
        analyze(_payload(base_wins=1))


def test_analysis_rejects_rate_that_cannot_be_recovered_as_an_action_count():
    payload = _payload()
    payload["rows"]["model +GRPO"]["illegal_rate"] = 0.123456789

    with pytest.raises(ValueError, match="integer action counts"):
        analyze(payload)


def test_analysis_rejects_ambiguous_action_denominator():
    payload = _payload()
    tuned = payload["rows"]["model +GRPO"]
    tuned["illegal_rate"] = 1 / 500
    tuned["tag_ok_rate"] = 499 / 500
    tuned["repeat_rate"] = 1 / 500
    tuned["absent_reuse_rate"] = 1 / 2
    tuned["green_break_rate"] = 1 / 4

    with pytest.raises(ValueError, match="unique integer action denominator"):
        analyze(payload)
