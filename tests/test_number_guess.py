"""煙霧環境（猜數字）：解析、遊戲流程、獎勵、二分搜尋必勝。"""

import pytest

from wordle_rl.env.number_guess import (
    MAX_TURNS,
    NumberGuessGame,
    number_guess_reward,
    parse_number_guess,
)


def test_parse_tag_ok():
    assert parse_number_guess("try <guess>50</guess>") == (50, "tag_ok")


def test_parse_out_of_range_tag_rejected():
    n, outcome = parse_number_guess("<guess>150</guess>")
    assert n is None and outcome == "no_parse"


def test_parse_fallback_last_number():
    assert parse_number_guess("maybe 30, no, 42") == (42, "fallback_word")


def test_parse_no_parse():
    assert parse_number_guess("hmm") == (None, "no_parse")


def test_game_flow_and_feedback_direction():
    g = NumberGuessGame(answer=42)
    assert not g.step("<guess>50</guess>")
    assert g.turns[-1]["feedback"] == "LOWER"      # 秘密數比 50 小
    assert not g.step("<guess>30</guess>")
    assert g.turns[-1]["feedback"] == "HIGHER"
    done = g.step("<guess>42</guess>")
    assert done and g.won
    s = g.stats()
    assert s["win"] == 1.0 and s["turns_used"] == 3 and s["num_illegal"] == 0


def test_game_messages_render_feedback():
    g = NumberGuessGame(answer=42)
    g.step("<guess>50</guess>")
    msgs = g.build_messages()
    assert msgs[2]["content"] == "<guess>50</guess>"
    assert "LOWER than 50" in msgs[3]["content"]
    assert "Turn 2 of 7" in msgs[3]["content"]


def test_repeat_and_illegal_tracking():
    g = NumberGuessGame(answer=99)
    g.step("<guess>50</guess>")
    g.step("<guess>50</guess>")
    g.step("no idea")
    s = g.stats()
    assert s["num_repeats"] == 1 and s["num_illegal"] == 1


def test_loss_after_max_turns():
    g = NumberGuessGame(answer=1)
    done = False
    for _ in range(MAX_TURNS):
        done = g.step("<guess>100</guess>")
    assert done and not g.won and g.stats()["turns_used"] == MAX_TURNS


def test_reward_values():
    assert number_guess_reward({"win": 1.0, "turns_used": 1, "num_illegal": 0, "num_repeats": 0}) == pytest.approx(8.0)
    assert number_guess_reward({"win": 1.0, "turns_used": 7, "num_illegal": 0, "num_repeats": 0}) == pytest.approx(5.0)
    assert number_guess_reward({"win": 0.0, "turns_used": 7, "num_illegal": 7, "num_repeats": 0}) == pytest.approx(-14.0)
    assert number_guess_reward({"win": 0.0, "turns_used": 7, "num_illegal": 0, "num_repeats": 2}) == pytest.approx(-4.0)


@pytest.mark.parametrize("answer", [1, 2, 37, 50, 63, 99, 100])
def test_binary_search_always_wins_within_7(answer):
    g = NumberGuessGame(answer=answer)
    lo, hi = 1, 100
    while not g.done:
        mid = (lo + hi) // 2
        g.step(f"<guess>{mid}</guess>")
        fb = g.turns[-1]["feedback"]
        if fb == "HIGHER":
            lo = mid + 1
        elif fb == "LOWER":
            hi = mid - 1
    assert g.won and len(g.turns) <= 7
