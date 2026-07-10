"""訓練 preset：SMOKE（猜數字煙霧）/ FULL（Wordle，A100 40GB）。

量級依 PLAN.md 超參表；per_turn × max_turns 必須 ≤ max_completion_length
（TRL 多輪模式下 max_completion_length 是「整局」token 上限——官方範例同）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TrainPreset:
    name: str
    game: str                      # "wordle" | "number"
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # 最佳化
    learning_rate: float = 1e-5
    beta: float = 0.0              # KL 係數；FULL 用 0.01（PEFT 下 ref = 關 adapter，無額外顯存）
    warmup_steps: int = 10
    # GRPO
    num_generations: int = 8       # 同答案一組
    per_device_batch: int = 8      # = num_generations → 1 組/micro-batch
    grad_accum: int = 1
    max_prompt_length: int = 512
    max_completion_length: int = 512   # 整局 token 預算
    per_turn_max_tokens: int = 64
    temperature: float = 1.0       # rollout 探索溫度（評測另走 greedy 路徑）
    top_p: float = 1.0
    game_max_turns: int = 7
    # 排程
    max_steps: int = 60
    checkpoint_minutes: int = 10
    save_total_limit: int = 3
    sample_every_steps: int = 25
    dataset_rows: int = 512
    # vLLM colocate
    vllm_gpu_memory_utilization: float = 0.25
    seed: int = 42

    def __post_init__(self):
        if self.game_max_turns * self.per_turn_max_tokens > self.max_completion_length:
            raise ValueError(
                f"整局預算不足：{self.game_max_turns}×{self.per_turn_max_tokens} > "
                f"{self.max_completion_length}（會在局中截斷）"
            )
        if self.per_device_batch % self.num_generations != 0:
            raise ValueError("per_device_batch 必須是 num_generations 的倍數（組不可跨 micro-batch）")


SMOKE = TrainPreset(
    name="smoke",
    game="number",
    beta=0.0,
    max_completion_length=512,
    per_turn_max_tokens=64,
    game_max_turns=7,          # number_guess 上限 7 回合
    max_steps=60,
    checkpoint_minutes=10,
    sample_every_steps=25,
    dataset_rows=512,
)

FULL = TrainPreset(
    name="full",
    game="wordle",
    beta=0.01,
    grad_accum=2,              # 2 組 = 16 局/optimizer step
    max_completion_length=1024,
    per_turn_max_tokens=160,
    game_max_turns=6,
    max_steps=400,             # --max-hours 8 兜底；sec/step 於 M3.2 實測校正
    checkpoint_minutes=30,
    sample_every_steps=50,
    dataset_rows=2048,
)

PRESETS = {"smoke": SMOKE, "full": FULL}


def get_preset(name: str, **overrides) -> TrainPreset:
    if name not in PRESETS:
        raise ValueError(f"未知 preset：{name!r}（可用：{sorted(PRESETS)}）")
    preset = PRESETS[name]
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(preset, **clean) if clean else preset
