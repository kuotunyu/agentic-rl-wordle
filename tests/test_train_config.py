"""config presets 與 train.py 的純函數部分（不碰 TRL）。"""

import json

import pytest

from wordle_rl.config import FULL, SMOKE, get_preset
from wordle_rl.train import build_game_wiring, find_resume_checkpoint


def test_presets_budget_invariant():
    assert SMOKE.game_max_turns * SMOKE.per_turn_max_tokens <= SMOKE.max_completion_length
    assert FULL.game_max_turns * FULL.per_turn_max_tokens <= FULL.max_completion_length


def test_full_preset_values_match_plan():
    assert FULL.game == "wordle" and FULL.num_generations == 8
    assert FULL.beta == pytest.approx(0.01)
    assert FULL.max_completion_length == 1024 and FULL.per_turn_max_tokens == 160
    assert FULL.checkpoint_minutes == 30 and FULL.sample_every_steps == 50


def test_get_preset_override_and_validation():
    p = get_preset("smoke", max_steps=5, beta=0.02)
    assert p.max_steps == 5 and p.beta == 0.02
    with pytest.raises(ValueError):
        get_preset("nope")
    with pytest.raises(ValueError):
        get_preset("full", per_turn_max_tokens=1024)  # 6×1024 > 1024 預算爆炸


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
