# 階段 2 選型結論：多輪 GRPO 訓練器

**結論：主案 = TRL 1.8 GRPOTrainer + 自訂 `rollout_func` + in-process 純 Python 環境；
備援 = OpenPipe ART LocalBackend；verifiers 僅作環境/獎勵設計參照，不用其訓練路徑。**

選型標準依規格：「能跑 > 優雅」。原規格的評估順序是 (a) verifiers → (b) ART → (c) TRL 自製，
但 2026-07-10 的網路查證（5 路平行研究，來源見下）顯示生態已大幅變動，據此重排。

## 各候選現況（2026-07-10 查證）

### verifiers（原 willccbb/verifiers → PrimeIntellect-ai/verifiers）— 不採用
- PyPI v0.1.14（2026-05-07）。定位已從「RL 訓練庫」轉向「環境 + 評測庫」。
- **自帶 GRPOTrainer 已於 v0.1.7（2025-11）棄用**；後繼 RLTrainer 被拆到 `verifiers-rl`
  套件——該套件**未發佈上 PyPI**（import shim 直接拋錯），main 分支已整包刪除。
- 官方訓練路徑改為 prime-rl：三程序（inference/orchestrator/trainer）、雙 GPU 導向、
  需 uv + Python 3.12，**查無任何 Colab 單卡先例**。對「單張 Colab A100 + 背景執行」
  的本專案約束實質 blocked。
- 保留價值：其 repo 內建 Wordle 環境的 rubric（correct 1.0 / 0.2×新綠 + 0.1×新黃 /
  format 0.2）是我們 shaping 量級的參照之一。

### OpenPipe ART（openpipe-art 0.5.18）— 備援
- 活躍維護，但重心轉向 serverless（W&B Serverless RL）；LocalBackend（vLLM + Unsloth
  LoRA、單 GPU 交替推理/訓練）仍是一級公民，官方 tic-tac-toe / temporal-clue notebook
  以 Qwen2.5-3B 在免費 T4 上示範 → A100 40GB 餘裕充足。
- API 契合度最高：`art.TrajectoryGroup(rollout(model, scenario) for _ in range(8))`
  天生就是「同 scenario 8 條軌跡一組」的 GRPO 語義。
- 風險：官方 notebook pin `openpipe-art[backend]==0.4.11 + vllm==0.9.2`，與 PyPI 0.5.18
  嚴重撕裂；LocalBackend 有多個未修 LoRA reload bug（#661/#678/#651/#469）；
  Google Drive 續跑需自己黏（.art 目錄手動同步）。
- 若 TRL spike 失敗的切換路徑：只需重寫 `wordle_rl/rollout.py`（本專案唯一 TRL 接觸面），
  以 tic_tac_toe notebook 為模板，pin 0.4.11 全家桶。

### TRL v1.8.0（2026-07-09）— 主案
- **官方就有 Wordle 多輪 GRPO 範例**：docs 的 openenv 頁 +
  `examples/scripts/openenv/wordle.py` + `examples/notebooks/openenv_wordle_grpo.ipynb`
  （Qwen3-1.7B、單 GPU vLLM colocate、num_generations=4、max_completion_length=1024）。
- 單 GPU colocate 是文件化的預設路徑：`GRPOConfig(use_vllm=True, vllm_mode="colocate",
  vllm_gpu_memory_utilization≈0.25, vllm_enable_sleep_mode=True)`。
- `rollout_func(prompts, trainer) -> dict`（實驗性）：必要鍵 prompt_ids / completion_ids /
  logprobs，**其他鍵轉發給 reward functions**；在 server 與 colocate 模式皆可用；
  收到的 prompts 未依 num_generations 重複（rollout 自行回 G 條/prompt）。
- assistant-only loss 採「結構性遮罩」：只有模型生成 token 進 completion_ids，環境回饋
  只進重新渲染的 prompt 側。這同時繞開 Qwen2.5 模板缺 `{% generation %}`、
  `return_assistant_tokens_mask` 靜默失效的坑。
- 跨回合 logprobs 串接 = 取樣時值的 importance-sampling 近似——官方範例同一作法。
- 風險與對策：rollout_func 屬實驗性（簽名歷史上改過兩次）→ **精確 pin trl==1.8.0**、
  TRL 接觸面全部關進 `rollout.py` + `train.py` 兩檔；v1.0（2026-03）是 breaking release，
  任何 v0.2x 時代教學都不可直接照抄。

## 本專案的兩個關鍵實作決策

1. **答案不進 dataset、不進 prompt**：rollout_func 內部用 seeded sampler 對每個 prompt
   槽位取一個隱藏答案、跑 8 局共用——GRPO 只要求「同組同答案」，不要求答案可從 prompt
   還原。這消滅了「game-id 埋 prompt」的協定漂移風險，dataset 只是排程載體（內容全同）。
2. **獎勵雙 preset**：HF 官方在其 Wordle 範例的結論是「純輸贏獎勵優於 shaped」；
   我們預設 SHAPED（規格版，且組內變異能緩解早期全敗的零梯度問題），保留
   `--reward binary` 做 A/B。零變異組占比進 metrics.jsonl 監控。

## M2.1 Spike（60 分鐘 timebox）檢核清單

在 Colab A100 執行 `python scripts/spike_trl.py`，逐項驗證：

| # | 驗證點 | 狀態 |
|---|---|---|
| A2 | `trl.experimental.openenv.generate_rollout_completions` 在 colocate 可用、接受 max_tokens/stop 覆寫 | ⬜ 待跑 |
| B | rollout_func 收到未重複的 prompts、回 G 條/prompt 被正確分組 | ⬜ 待跑 |
| C | 額外欄位（win/turns_used/…）轉發進 reward_fn kwargs | ⬜ 待跑 |
| D | 一個乾淨 optimizer step、loss 有限、checkpoint 可寫 | ⬜ 待跑 |
| — | 可用版本三元組 (torch, vllm, trl) 回填 requirements-colab.txt | ⬜ 待跑 |

**Spike 結果**：（跑完回填：PASS/FAIL、實際版本、旗標差異、修改點）

## 研究來源
- TRL releases / GRPOTrainer docs / openenv docs（rollout_func 契約、colocate、Wordle 範例）：
  github.com/huggingface/trl、huggingface.co/docs/trl/main/en/openenv、/grpo_trainer
- rollout_func 歷史：TRL v0.25.1 vs v0.27.1 docs、issue #5121 / PR #5122
- Qwen2.5 assistant mask 缺陷：transformers issue #34172
- verifiers 現況：github.com/PrimeIntellect-ai/verifiers（v0.1.14 tag 的 packages/、docs/training.md）
- prime-rl 單卡：PR #971；examples/wordle（2 GPU + SFT 暖身）
- ART 現況：github.com/OpenPipe/ART（releases、art-notebooks、issues #661/#678/#651/#469）
- colocate 顯存機制：HF blog「No GPU left behind: co-located vLLM in TRL」
