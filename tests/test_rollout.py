"""rollout 引擎（lockstep / 串接 / 預算 / 轉發）——用 fake generate 全程 CPU 測試。"""

import sys
import types

import pytest

from wordle_rl.env.number_guess import NumberGuessGame
from wordle_rl.games import WordleGame
from wordle_rl.rollout import (
    RolloutBuffers,
    SpikeValidationError,
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
    # 模擬 TRL RepeatSampler 已展開好的批次：每個底層 prompt 連續重複 num_generations 次
    # （trl/trainer/utils.py RepeatSampler 文件字串 mini_repeat_count 語意確認）——
    # rollout_func 不能再自己乘一次，一一對應才是正確契約。
    prompts = ["p0"] * 4 + ["p1"] * 4
    out = fn(prompts, FakeTrainer())

    n = len(prompts)
    for key in ("prompt_ids", "completion_ids", "logprobs", "win", "turns_used", "answer_used"):
        assert len(out[key]) == n, key
    # 同組（連續 4 個）共用同一答案；跨組答案不同（sampler 每 epoch 全覆蓋）
    assert len(set(out["answer_used"][:4])) == 1
    assert len(set(out["answer_used"][4:])) == 1
    assert set(out["answer_used"]) == {"crane", "slate"}
    # buffers 有記錄
    snap = buffers.snapshot_metrics()
    assert "rollout/win_rate" in snap
    assert buffers.sample_transcripts(2)


def test_make_rollout_func_rejects_non_multiple_of_num_generations():
    """TRL 契約：len(prompts) 必須是 num_generations 的整數倍（RepeatSampler 保證此不變量）；
    不成立時要立刻報錯，不能默默算出數量錯誤的批次去撞 TRL 內部的 shuffle/reshape
    （M2.1 spike 實際撞到的 CUDA index-out-of-bounds 根因）。
    """
    fn = make_rollout_func(
        game_factory=lambda ans: WordleGame(ans, LEGAL),
        answers=["crane"],
        num_generations=4,
        per_turn_max_tokens=32,
        max_total_tokens=256,
        seed=1,
    )
    with pytest.raises(SpikeValidationError, match="num_generations"):
        fn(["p0", "p1"], FakeTrainer())  # 2 不是 4 的倍數


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
    # 模擬 RepeatSampler 已展開好的一組（2 個一樣的 prompt = num_generations）
    out = fn(["p0", "p0"], FakeTrainer())
    assert len(out["win"]) == 2
    # 答案 50 → 一回合全勝；答案 51 → 永遠猜 50 到 7 回合敗
    assert out["win"] in ([1.0, 1.0], [0.0, 0.0])


def _install_fake_trl_openenv(fn):
    """把假的 trl.experimental.openenv.generate_rollout_completions 塞進 sys.modules，
    這樣不用真的安裝 trl 也能測 _trl_generate_fn 的欄位解析邏輯（M2.1 spike 實錄的回歸測試：
    generate_rollout_completions 回傳的欄位名不對時絕不能靜默塞空 list）。
    """
    trl_mod = types.ModuleType("trl")
    experimental_mod = types.ModuleType("trl.experimental")
    openenv_mod = types.ModuleType("trl.experimental.openenv")
    openenv_mod.generate_rollout_completions = fn
    trl_mod.experimental = experimental_mod
    experimental_mod.openenv = openenv_mod
    sys.modules["trl"] = trl_mod
    sys.modules["trl.experimental"] = experimental_mod
    sys.modules["trl.experimental.openenv"] = openenv_mod


@pytest.fixture(autouse=False)
def fake_trl_module():
    saved = {k: sys.modules.get(k) for k in ("trl", "trl.experimental", "trl.experimental.openenv")}
    yield _install_fake_trl_openenv
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def test_trl_generate_fn_extracts_correct_fields(fake_trl_module):
    def fake_grc(trainer, prompts, **kwargs):
        return [
            {"completion_ids": [1, 2, 3], "logprobs": [-0.1, -0.2, -0.3], "text": "<guess>50</guess>"}
            for _ in prompts
        ]

    fake_trl_module(fake_grc)
    from wordle_rl.rollout import _trl_generate_fn

    gen = _trl_generate_fn(
        FakeTrainer(), per_turn_max_tokens=16, temperature=1.0, top_p=1.0, stop=("</guess>",)
    )
    out = gen([[{"role": "user", "content": "hi"}]])
    assert len(out) == 1
    assert out[0].token_ids == (1, 2, 3)
    assert out[0].logprobs == (-0.1, -0.2, -0.3)
    assert out[0].text == "<guess>50</guess>"


def test_trl_generate_fn_missing_fields_fails_loud_not_silent(fake_trl_module):
    """迴歸測試：generate_rollout_completions 回傳欄位名對不上時必須立刻報錯附可用欄位，
    絕不能靜默塞空 list（那會讓 GRPO 拿零長度 completion 去算 loss，在 CUDA 底層炸出
    難懂的 index-out-of-bounds，而不是在這裡就講清楚——M2.1 spike 實際撞到的 bug）。
    """
    def fake_grc(trainer, prompts, **kwargs):
        return [{"token_ids": [1, 2, 3], "logprob": [-0.1]} for _ in prompts]  # 欄位名故意錯

    fake_trl_module(fake_grc)
    from wordle_rl.rollout import SpikeValidationError, _trl_generate_fn

    gen = _trl_generate_fn(
        FakeTrainer(), per_turn_max_tokens=16, temperature=1.0, top_p=1.0, stop=("</guess>",)
    )
    with pytest.raises(SpikeValidationError, match="token_ids"):
        gen([[{"role": "user", "content": "hi"}]])
