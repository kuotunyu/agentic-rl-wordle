"""訓練 preset：SMOKE（猜數字煙霧）/ FULL（Wordle，A100 40GB）。

量級依 PLAN.md 超參表。max_completion_length 是「整局」token 上限，必須同時容納：
(a) 模型真實生成的 token（game_max_turns × per_turn_max_tokens，= 送進 rollout 引擎的
    原始生成預算，控制何時停止再生成下一回合）；(b) 回合間插入的環境回饋文字（M2.1
    spike 修正：env_mask 機制要求 completion_ids 是整段連續序列，含回饋文字才能讓 TRL
    重算 logprobs 時對得上真實上下文——見 rollout.py 模組說明），用
    feedback_overhead_per_turn 估算每回合大約需要多少額外空間。
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
    max_completion_length: int = 1024  # 整局 token 預算（含回合間插入的回饋文字）
    per_turn_max_tokens: int = 64
    feedback_overhead_per_turn: int = 48  # 每回合回饋文字+chat template wrapper 的估計 token 數
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

    @property
    def raw_generation_budget(self) -> int:
        """送進 rollout 引擎的原始生成預算（不含回饋文字）——控制何時停止再生成下一回合。"""
        return self.game_max_turns * self.per_turn_max_tokens

    def __post_init__(self):
        needed = self.game_max_turns * (self.per_turn_max_tokens + self.feedback_overhead_per_turn)
        if needed > self.max_completion_length:
            raise ValueError(
                f"整局預算不足：{self.game_max_turns}×({self.per_turn_max_tokens}+"
                f"{self.feedback_overhead_per_turn}) = {needed} > {self.max_completion_length}"
                "（含回饋文字後可能在局中截斷）"
            )
        if self.per_device_batch % self.num_generations != 0:
            raise ValueError("per_device_batch 必須是 num_generations 的倍數（組不可跨 micro-batch）")


SMOKE = TrainPreset(
    name="smoke",
    game="number",
    beta=0.0,
    max_completion_length=1024,  # 7×(64+48)=784 的原始+回饋預算，留餘裕到 1024
    per_turn_max_tokens=64,
    game_max_turns=7,          # number_guess 上限 7 回合
    # 60 步（480 局）在 M2.3 gate 實測時雜訊蓋過訊號：win_rate 每步只從 8 條軌跡算出、
    # 粗顆粒度（0/8, 1/8, ...）震盪劇烈，60 步統計上看不出趨勢。人工核對 samples/
    # transcript 確認沒有 reward hacking、格式健康，判斷是量不夠而非機制壞掉，
    # 拉長到 200 步（1600 局）換取更有統計意義的曲線。
    max_steps=200,
    checkpoint_minutes=10,
    sample_every_steps=25,
    dataset_rows=512,
)

FULL = TrainPreset(
    name="full",
    game="wordle",
    beta=0.01,
    grad_accum=2,              # 2 組 = 16 局/optimizer step
    max_completion_length=1536,  # 6×(160+48)=1248 的原始+回饋預算，留餘裕到 1536
    per_turn_max_tokens=160,
    game_max_turns=6,
    max_steps=400,             # --max-hours 8 兜底；sec/step 於 M3.2 實測校正
    checkpoint_minutes=30,
    sample_every_steps=50,
    dataset_rows=2048,
    # M3.2 pilot 實測：預設 0.25 在 A100 40GB 上跑 5 步後穩定 OOM（訓練側記憶體
    # 逐步爬升吃滿整卡）——調低給訓練側多一點餘裕，配合 train.py 的
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments 修法一起用。
    vllm_gpu_memory_utilization=0.2,
)

PRESETS = {"smoke": SMOKE, "full": FULL}


def get_preset(name: str, **overrides) -> TrainPreset:
    if name not in PRESETS:
        raise ValueError(f"未知 preset：{name!r}（可用：{sorted(PRESETS)}）")
    preset = PRESETS[name]
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(preset, **clean) if clean else preset
