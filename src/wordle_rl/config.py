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
    per_device_batch: int = 8      # micro-batch 大小；與組大小「獨立」——TRL 的組 advantage 在
                                   # 生成階段就算完才切 micro-batch，唯一約束見 __post_init__
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
        # TRL 1.8 實碼驗證（grpo_config.py:1088-1094）：整除約束掛在「生成批」
        # generation_batch_size = per_device_batch × grad_accum（× world_size）上，
        # 不是 per_device_batch 本身。組內 advantage 於生成時對整個生成批算完
        # 才切 micro-batch（grpo_trainer.py _prepare_inputs），組跨 micro-batch 是
        # TRL 的正常運作模式——舊版檢查「per_device_batch % num_generations」是
        # 過度限制，會擋掉合法的 4×4 配置（首次 FULL 訓練 OOM 修正需要它）。
        if (self.per_device_batch * self.grad_accum) % self.num_generations != 0:
            raise ValueError(
                "生成批（per_device_batch × grad_accum）必須是 num_generations 的倍數"
                "（TRL 對 generation_batch_size 的整除約束）"
            )


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
    # ---- OOM 根因修正（首次 FULL 訓練 6 連炸的實測診斷）----
    # 崩潰點永遠在 loss.backward() 要求 7.2~7.36 GiB：反推 = micro-batch 8 ×
    # 序列 ~1600（prompt + 含回饋的 completion 逼近 1536 上限）× 詞彙表 151,936
    # × 4 bytes 的 fp32 logits 級張量，數學誤差 ±2%。碎片化/洩漏/vLLM 增長等
    # 對立假說已逐一對照 log 排除（崩潰步數 14~159 隨機、reserved 未用僅 ~200MB、
    # 要求大小恆定）。修法：micro-batch 8→4、grad_accum 2→4——生成批仍是 16 局
    # = 2 組/optimizer step，TRL 的組 advantage 在生成時算完才切 micro-batch，
    # 訓練數學完全等價，但所有 batch 比例的張量峰值砍半（峰值估 ~44GB → ~27GB）。
    # 搭配 Colab A100「大量 RAM」開 80GB 卡（該開關切的是 40/80GB 顯卡）雙保險。
    per_device_batch=4,
    grad_accum=4,              # 4 micro-batch × 4 = 16 局/optimizer step（仍 2 組/步）
    max_completion_length=1536,  # 6×(160+48)=1248 的原始+回饋預算，留餘裕到 1536
    per_turn_max_tokens=160,
    game_max_turns=6,
    # 首次 FULL 實測 ~4.5-5 秒/optimizer step（原估 60-120 秒嚴重高估——rollout 比
    # 想像便宜，模型每局實際只生成 ~50 token）。400 步約半小時就結束，對不起開
    # 8 小時的機器；校正到 3000 步（約 4-5 小時），--max-hours 8 仍是硬兜底。
    max_steps=3000,
    # 六次 OOM 全部從第 0 步重來的原因：崩潰都發生在第一個 checkpoint（30 分鐘）
    # 之前，--resume auto 無檔可續。縮到 10 分鐘讓自動重啟真的有意義。
    checkpoint_minutes=10,
    sample_every_steps=50,
    dataset_rows=2048,
    # vllm_gpu_memory_utilization 歷程：0.25 → 第 5 步 OOM；0.2 → 第 49 步 OOM；
    # 0.15 + 上面的 micro-batch 砍半 = 根因修正後的最終組合。
    vllm_gpu_memory_utilization=0.15,
)

PRESETS = {"smoke": SMOKE, "full": FULL}


def get_preset(name: str, **overrides) -> TrainPreset:
    if name not in PRESETS:
        raise ValueError(f"未知 preset：{name!r}（可用：{sorted(PRESETS)}）")
    preset = PRESETS[name]
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(preset, **clean) if clean else preset
