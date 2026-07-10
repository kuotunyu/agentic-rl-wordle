"""rollout 引擎（lockstep / 串接 / 預算 / 轉發）——用 fake generate 全程 CPU 測試。"""

import pytest

from wordle_rl.env.number_guess import NumberGuessGame
from wordle_rl.games import WordleGame
from wordle_rl.rollout import (
    RolloutBuffers,
    TurnGen,
    make_answer_sampler,
    make_rollout_func,
    make_trl_reward_fn,
    play_batch_episodes,
)

LEGAL = frozenset({"crane", "slate", "paper", "arise", "route"})


def gen_of(texts_by_call):
    """依呼叫序回傳腳本文字；token ids/logprobs 用長度可驗證的假資料。"""
    calls = {"n": 0}

    def generate(chats):
        texts = texts_by_call[calls["n"]]
        calls["n"] += 1
        assert len(texts) == len(chats), "腳本與 active 數不符"
        out = []
        for t in texts:
            ids = tuple(range(len(t)))  # 假 token：一字元一 token
            lps = tuple(-0.1 for _ in ids)
            out.append(TurnGen(token_ids=ids, logprobs=lps, text=t))
        return out

    generate.calls = calls
    return generate


def test_lockstep_winner_drops_out():
    g1 = WordleGame("crane", LEGAL)  # 第一回合就中
    g2 = WordleGame("slate", LEGAL)
    generate = gen_of(
        [
            ["<guess>crane</guess>", "<guess>crane</guess>"],  # g1 勝、g2 得回饋
            ["<guess>slate</guess>"],                          # 只剩 g2
        ]
    )
    rollouts = play_batch_episodes(generate, [g1, g2], per_turn_max_tokens=64)
    assert rollouts[0].stats["win"] == 1.0 and rollouts[0].stats["turns_used"] == 1
    assert rollouts[1].stats["win"] == 1.0 and rollouts[1].stats["turns_used"] == 2
    assert generate.calls["n"] == 2


def test_completion_concat_and_logprob_alignment():
    g = WordleGame("crane", LEGAL)
    t1, t2 = "<guess>slate</guess>", "<guess>crane</guess>"
    generate = gen_of([[t1], [t2]])
    (r,) = play_batch_episodes(generate, [g], per_turn_max_tokens=64)
    assert len(r.completion_ids) == len(t1) + len(t2)
    assert len(r.logprobs) == len(r.completion_ids)


def test_budget_truncation_marks_episode():
    g = WordleGame("crane", LEGAL)
    generate = gen_of([["<guess>slate</guess>"]])
    # 預算只夠一回合（每回合最多 30 token，總預算 40）
    (r,) = play_batch_episodes(
        generate, [g], per_turn_max_tokens=30, max_total_tokens=40
    )
    assert r.budget_truncated and r.stats["win"] == 0.0
    assert generate.calls["n"] == 1


def test_token_logprob_mismatch_raises():
    g = WordleGame("crane", LEGAL)

    def bad_generate(chats):
        return [TurnGen(token_ids=(1, 2, 3), logprobs=(-0.1,), text="<guess>crane</guess>")]

    with pytest.raises(RuntimeError, match="長度不齊"):
        play_batch_episodes(bad_generate, [g], per_turn_max_tokens=64)


def test_answer_sampler_deterministic_and_covering():
    s1 = make_answer_sampler(["a", "b", "c"], seed=1)
    s2 = make_answer_sampler(["a", "b", "c"], seed=1)
    seq1 = [s1() for _ in range(6)]
    seq2 = [s2() for _ in range(6)]
    assert seq1 == seq2
    assert set(seq1[:3]) == {"a", "b", "c"}  # 每輪 epoch 全覆蓋
    assert set(seq1[3:]) == {"a", "b", "c"}


class FakeTokenizer:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "\n".join(f"{m['role']}:{m['content']}" for m in msgs) + "\nassistant:"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 97 for c in text[:16]]}

    def decode(self, ids, skip_special_tokens=False):
        return "<decoded>"


class FakeTrainer:
    processing_class = FakeTokenizer()


def test_make_rollout_func_group_semantics():
    buffers = RolloutBuffers()

    def fake_gen_factory(trainer, **kw):
        def generate(chats):
            # 每局永遠猜 crane（對 crane 答案一回合就中）
            return [
                TurnGen(token_ids=(1, 2), logprobs=(-0.5, -0.5), text="<guess>crane</guess>")
                for _ in chats
            ]

        return generate

    fn = make_rollout_func(
        game_factory=lambda ans: WordleGame(ans, LEGAL),
        answers=["crane", "slate"],
        num_generations=4,
        per_turn_max_tokens=32,
        max_total_tokens=256,
        seed=7,
        buffers=buffers,
        generate_fn_factory=fake_gen_factory,
    )
    out = fn(["p0", "p1"], FakeTrainer())

    n = 2 * 4
    for key in ("prompt_ids", "completion_ids", "logprobs", "win", "turns_used", "answer_used"):
        assert len(out[key]) == n, key
    # 同組共用同一答案；兩組答案不同（sampler 每 epoch 全覆蓋）
    assert len(set(out["answer_used"][:4])) == 1
    assert len(set(out["answer_used"][4:])) == 1
    assert set(out["answer_used"]) == {"crane", "slate"}
    # buffers 有記錄
    snap = buffers.snapshot_metrics()
    assert "rollout/win_rate" in snap
    assert buffers.sample_transcripts(2)


def test_make_trl_reward_fn_wordle_shaped_and_zero_variance():
    buffers = RolloutBuffers()
    fn = make_trl_reward_fn("wordle", "shaped", num_generations=2, buffers=buffers)
    kwargs = {
        "win": [1.0, 1.0, 0.0, 0.0],
        "turns_used": [1, 1, 6, 6],
        "num_illegal": [0, 0, 6, 6],
        "num_violation_turns": [0, 0, 0, 0],
        "num_repeats": [0, 0, 0, 0],
        "total_new_greens": [5, 5, 0, 0],
        "total_new_presence": [5, 5, 0, 0],
        "answer_used": ["crane", "crane", "slate", "slate"],
    }
    rewards = fn(completions=None, **kwargs)
    assert rewards[0] == pytest.approx(16.5)
    assert rewards[2] == pytest.approx(-12.0)
    # 兩組組內皆同值 → zero-variance 比例 = 1.0
    assert buffers.snapshot_metrics()["reward/zero_variance_group_frac"] == pytest.approx(1.0)


def test_make_trl_reward_fn_number():
    fn = make_trl_reward_fn("number", "shaped", num_generations=2)
    rewards = fn(
        completions=None,
        win=[1.0, 0.0],
        turns_used=[1, 7],
        num_illegal=[0, 7],
        num_repeats=[0, 0],
    )
    assert rewards == [pytest.approx(8.0), pytest.approx(-14.0)]


def test_rollout_func_works_with_number_game():
    def fake_gen_factory(trainer, **kw):
        state = {"n": 0}

        def generate(chats):
            state["n"] += 1
            return [
                TurnGen(token_ids=(1,), logprobs=(-0.2,), text="<guess>50</guess>")
                for _ in chats
            ]

        return generate

    fn = make_rollout_func(
        game_factory=lambda ans: NumberGuessGame(answer=ans),
        answers=[50, 51],
        num_generations=2,
        per_turn_max_tokens=16,
        max_total_tokens=128,
        seed=3,
        generate_fn_factory=fake_gen_factory,
    )
    out = fn(["p0"], FakeTrainer())
    assert len(out["win"]) == 2
    # 答案 50 → 一回合全勝；答案 51 → 永遠猜 50 到 7 回合敗
    assert out["win"] in ([1.0, 1.0], [0.0, 0.0])
