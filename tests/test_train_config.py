"""config presets 與 train.py 的純函數部分（不碰 TRL）。"""

import json

import pytest

from wordle_rl.config import FULL, SMOKE, get_preset
from wordle_rl.train import build_game_wiring, find_resume_checkpoint


def test_presets_budget_invariant():
    # 含回饋文字估計值的完整預算必須 <= max_completion_length（M2.1 spike 修正後的公式）
    for p in (SMOKE, FULL):
        needed = p.game_max_turns * (p.per_turn_max_tokens + p.feedback_overhead_per_turn)
        assert needed <= p.max_completion_length
        # raw_generation_budget（不含回饋）嚴格小於含回饋的完整預算
        assert p.raw_generation_budget == p.game_max_turns * p.per_turn_max_tokens
        assert p.raw_generation_budget < p.max_completion_length


def test_full_preset_values_match_plan():
    assert FULL.game == "wordle" and FULL.num_generations == 8
    assert FULL.beta == pytest.approx(0.01)
    assert FULL.max_completion_length == 1536 and FULL.per_turn_max_tokens == 160
    # 首次 FULL 訓練 OOM 修正：micro-batch 4×accum 4 = 生成批 16 = 2 組/步不變；
    # checkpoint 縮到 10 分鐘（六次崩潰全發生在首個 30 分 checkpoint 前，resume 無檔可續）
    assert FULL.per_device_batch == 4 and FULL.grad_accum == 4
    assert FULL.per_device_batch * FULL.grad_accum == 2 * FULL.num_generations
    assert FULL.checkpoint_minutes == 10 and FULL.sample_every_steps == 50
    assert FULL.max_steps == 3000  # 實測 ~5 秒/步後的校正值（原 400 步半小時就跑完）


def test_get_preset_override_and_validation():
    p = get_preset("smoke", max_steps=5, beta=0.02)
    assert p.max_steps == 5 and p.beta == 0.02
    with pytest.raises(ValueError):
        get_preset("nope")
    with pytest.raises(ValueError):
        get_preset("full", per_turn_max_tokens=1024)  # 6×(1024+48) 遠超 1536 預算爆炸
    # 生成批整除約束（TRL generation_batch_size % num_generations == 0）：
    # micro-batch 可以小於組大小（FULL 的 4×4=16 ✓），但生成批不是組的倍數要擋下
    with pytest.raises(ValueError):
        get_preset("smoke", per_device_batch=4)  # 4×1 = 4 不是 8 的倍數
    ok = get_preset("smoke", per_device_batch=4, grad_accum=2)  # 4×2 = 8 ✓
    assert ok.per_device_batch * ok.grad_accum == ok.num_generations


def test_find_resume_checkpoint(tmp_path):
    assert find_resume_checkpoint(tmp_path) is None
    good = tmp_path / "checkpoint-100"
    good.mkdir()
    (good / "trainer_state.json").write_text(json.dumps({"global_step": 100}))
    newer_broken = tmp_path / "checkpoint-200"
    newer_broken.mkdir()
    (newer_broken / "trainer_state.json").write_text("{corrupt", encoding="utf-8")
    (tmp_path / "checkpoint-abc").mkdir()  # 非法名稱直接忽略
    # 最新的壞掉 → 退到次新的完好 checkpoint
    assert find_resume_checkpoint(tmp_path) == str(good)


def test_build_game_wiring_wordle_uses_train_split_only():
    from wordle_rl.words import get_splits

    factory, answers, initial = build_game_wiring(FULL)
    splits = get_splits(seed=42)
    assert set(answers) == set(splits.train)
    assert not set(answers) & set(splits.eval_full)  # 紅線：eval 隔離
    assert [m["role"] for m in initial] == ["system", "user"]
    game = factory(answers[0])
    assert not game.done


def test_build_game_wiring_number():
    factory, answers, initial = build_game_wiring(SMOKE)
    assert answers == list(range(1, 101))
    assert "number" in initial[0]["content"].lower() or "1 and 100" in initial[0]["content"]
