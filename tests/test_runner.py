"""runner 對局迴圈 + 三種 agent 的行為測試（純 CPU，不碰 LLM backend）。"""

from wordle_rl.agents import HeuristicAgent, RandomAgent
from wordle_rl.env.wordle import ILLEGAL
from wordle_rl.metrics import aggregate
from wordle_rl.runner import run_episodes
from wordle_rl.words import get_splits, load_answers, load_legal


class ScriptedAgent:
    """依腳本逐回合出手（每局同步驟）。"""

    def __init__(self, script: list[str]):
        self.script = script
        self.calls = 0

    def act_batch(self, states):
        text = self.script[self.calls]
        self.calls += 1
        return [text for _ in states]


def test_scripted_win_and_records():
    legal = load_legal()
    agent = ScriptedAgent(
        [
            "let me try <guess>slate</guess>",
            "no tag here at all ???",  # no_parse → 浪費回合
            "<guess>slate</guess>",  # 重複
            "<guess>crane</guess>",
        ]
    )
    eps = run_episodes(agent, ["crane"], legal)
    ep = eps[0]
    assert ep.won and ep.turns_used == 4
    assert ep.num_illegal == 1
    assert ep.num_repeats == 1
    t2 = ep.turns[1]
    assert t2.parse_outcome == "no_parse" and t2.feedback == ILLEGAL
    assert ep.turns[2].is_repeat


def test_random_agent_always_legal():
    legal = load_legal()
    agent = RandomAgent(legal, seed=0)
    eps = run_episodes(agent, ["crane", "dolly"], legal)
    m = aggregate(eps)
    assert m.illegal_rate == 0.0
    assert m.tag_ok_rate == 1.0
    for ep in eps:
        assert ep.turns_used == 6 or ep.won


def test_random_agent_deterministic_by_seed():
    legal = load_legal()
    e1 = run_episodes(RandomAgent(legal, seed=7), ["crane"], legal)
    e2 = run_episodes(RandomAgent(legal, seed=7), ["crane"], legal)
    assert [t.guess for t in e1[0].turns] == [t.guess for t in e2[0].turns]


def test_heuristic_agent_wins_most_small_sample():
    """啟發式在 20 個 eval 詞上應該大殺特殺（>=80% 勝率）且零違規零非法。"""
    legal = load_legal()
    answers = load_answers()
    words = list(get_splits().eval_200[:20])
    eps = run_episodes(HeuristicAgent(answers), words, legal)
    m = aggregate(eps)
    assert m.illegal_rate == 0.0
    assert m.win_rate >= 0.8
    # 過濾式啟發式永遠猜一致候選 → 不可能違規
    assert (m.absent_reuse_rate or 0.0) == 0.0
    assert (m.green_break_rate or 0.0) == 0.0
    assert m.repeat_rate == 0.0


def test_heuristic_deterministic():
    legal = load_legal()
    answers = load_answers()
    e1 = run_episodes(HeuristicAgent(answers), ["crane"], legal)
    e2 = run_episodes(HeuristicAgent(answers), ["crane"], legal)
    assert [t.guess for t in e1[0].turns] == [t.guess for t in e2[0].turns]
