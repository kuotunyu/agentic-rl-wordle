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
  logprobs，**其他鍵轉發給 reward functions**；在 server 與 colocate 模式皆可用。
  **⚠️ 已被 spike 推翻的研究假設**：原以為「prompts 未依 num_generations 重複」，
  實測讀 trl 原始碼（`trl/trainer/utils.py` 的 `RepeatSampler`，
  `mini_repeat_count=self.num_generations`）證實**恰好相反**——TRL 自己就把每個
  dataset row 連續重複 num_generations 次才交給 rollout_func，我們不能再乘一次
  （細節見下方 M2.1 實測記錄）。
- assistant-only loss **不能**用「結構性排除」（只把生成 token 放進 completion_ids、
  回饋文字整段丟掉只留在下一輪 prompt）——這個原始設計已被 spike 證明是錯的：TRL
  訓練時要拿 `prompt_ids+completion_ids` 重新算「當下策略」logprobs，缺了回合間
  回饋的假上下文會讓重要性採樣比值崩潰到趨近零，梯度恆為 0（看似訓練有跑，實際
  什麼都沒學到）。正確做法是官方 `_tool_call_loop` 用的 **`env_mask`**：completion_ids
  整段連續（含回饋文字），額外回傳 `env_mask`（1=模型生成、0=環境插入）讓 TRL 只對
  mask=1 位置算 loss，同時餵給重算 logprobs 的是模型真正見過的上下文。
- 跨回合同一回合內的 logprobs 是取樣時值（importance-sampling 近似）；回合間插入的
  回饋文字沒有真實 logprob，填 0.0（反正 mask=0，loss 不會用到）。
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

## M2.1 Spike（實測記錄，2026-07-10，Colab L4）

**結果：PASS。** 花了遠超 60 分鐘 timebox（版本漂移+一個核心邏輯 bug，逐一實測排除），
但每個問題都有真實環境證據支撐，不是憑空猜的；過程本身驗證了「先跑最小可行流程、
讓真實錯誤指路」這個 spike 策略是對的。

| # | 驗證點 | 狀態 |
|---|---|---|
| A2 | `trl.experimental.openenv.generate_rollout_completions` 在 colocate 可用、接受 max_tokens/stop 覆寫 | ✅ 通過（欄位名 completion_ids/logprobs/text 與假設一致，讀原始碼 `trl/experimental/openenv/utils.py` 確認） |
| B | rollout_func 收到的 prompts 語義 | ✅ 但假設錯誤已修正——見下方「已推翻的假設」 |
| C | 額外欄位（win/turns_used/…）轉發進 reward_fn kwargs | ✅ 通過 |
| D | 一個乾淨 optimizer step、loss 有限、checkpoint 可寫 | ✅ 通過（`grad_norm` 非零、`importance_sampling_ratio/mean` 落在 0.5~0.7 合理範圍、`reward/mean` 兩步內 1.31→4.19 有明顯進步） |
| — | 可用版本三元組回填 requirements-colab.txt | ✅ 已 pin |

**可用版本三元組**（已寫進 `requirements-colab.txt`）：
`torch==2.11.0+cu128`（Colab 內建，不覆蓋）｜`vllm==0.23.0`｜`trl==1.8.0`｜
`transformers==5.12.1`｜`peft==0.19.1`

### 依序踩到並修好的坑（真實環境證據，非推測）

1. **vllm 未 pin 抓到 0.24.0**：TRL 1.8.0 官方只支援 vllm 0.16.0–0.23.0，抓到最新版直接
   `ImportError: libcudart.so.13`（見下一條）。→ pin `vllm==0.23.0`。
2. **libcudart.so.13 找不到**：vllm 0.23.0 的編譯擴充套件 `vllm._C` 連到 CUDA13 runtime，
   pip 也確實裝了提供它的 `nvidia-cuda-runtime`（Colab 這批套件無 `-cu12` 尾綴＝cu13 版），
   但該 `.so` 沒進動態連結器預設搜尋路徑。**`os.environ["LD_LIBRARY_PATH"]` 在 process
   跑起來後修改對這個 process 自己的後續 import 沒有用**（glibc 只在 process 啟動當下讀
   一次）——第一版修法因此無效，改用 `ctypes.CDLL(path, mode=RTLD_GLOBAL)` 把 `.so`
   強制預先載入目前 process 才真的解決（`train.py` 的 `_fix_missing_cuda13_runtime_ld_path`）。
3. **`GRPOConfig.__init__() got an unexpected keyword argument 'max_prompt_length'`**：
   trl 1.8.0 的 GRPOConfig 已無此欄位（用自訂 rollout_func 時 TRL 不需要另外管 prompt
   長度）。用 `dataclasses.fields(GRPOConfig)` 現場列出真實欄位名確認，拿掉即可。
4. **torchao 版本太舊**：Colab 預裝 0.10.0，peft 的 LoRA dispatch 要求 >0.16.0
   （`peft/tuners/lora/torchao.py` 的 `is_torchao_available()` 檢查）。pin `torchao>=0.16.0`。
5. **核心邏輯 bug：rollout_func 對已展開的 prompts 又乘了一次 num_generations**（見上方
   「已推翻的假設」），導致回傳筆數是 TRL 預期的 num_generations 倍，在
   `shuffle_sequence_dict` 的 `permute`（`v[permutation]`）階段索引越界、CUDA
   `device-side assert triggered`。用 `CUDA_LAUNCH_BLOCKING=1` 拿到準確 Python traceback
   才定位到 `trl/trainer/utils.py:839 permute`，再讀 `RepeatSampler` 原始碼確認真相。
6. **核心邏輯 bug：多輪回饋文字被排除在 completion_ids 外，梯度恆為 0**（見上方
   assistant-only loss 說明）。改用 `env_mask` 機制，回傳整段連續 completion_ids +
   平行的 0/1 遮罩。連帶把 `max_completion_length` 拆成「原始生成預算」（控制何時停止
   再生成下一回合）與「含回饋文字的完整預算」（餵給 GRPOConfig）兩個獨立數字
   （`TrainPreset.raw_generation_budget` / `.max_completion_length`）。

### 已推翻的假設

原規格假設「prompts 是未依 num_generations 重複的原始切片」——**錯**。讀
`trl/trainer/utils.py` 的 `RepeatSampler`（建構參數 `mini_repeat_count=num_generations`，
文件字串範例直接畫出 `[0,0,0,1,1,1,2,2,2,...]` 這種連續重複模式）證實：TRL 自己就把
每個底層 dataset row 連續重複 num_generations 次才交給 rollout_func，`len(prompts)`
收到時已經等於 `generation_batch_size`（= num_generations 的整數倍）。`make_rollout_func`
現在對每個收到的 prompt 只產生一局，每 num_generations 個連續 prompt 共用一個抽樣答案，
並在開頭斷言 `len(prompts) % num_generations == 0`（不成立直接報錯，不默默算錯批次）。

## 研究來源
- TRL releases / GRPOTrainer docs / openenv docs（rollout_func 契約、colocate、Wordle 範例）：
  github.com/huggingface/trl、huggingface.co/docs/trl/main/en/openenv、/grpo_trainer
- rollout_func 歷史：TRL v0.25.1 vs v0.27.1 docs、issue #5121 / PR #5122
- Qwen2.5 assistant mask 缺陷：transformers issue #34172
- verifiers 現況：github.com/PrimeIntellect-ai/verifiers（v0.1.14 tag 的 packages/、docs/training.md）
- prime-rl 單卡：PR #971；examples/wordle（2 GPU + SFT 暖身）
- ART 現況：github.com/OpenPipe/ART（releases、art-notebooks、issues #661/#678/#651/#469）
- colocate 顯存機制：HF blog「No GPU left behind: co-located vLLM in TRL」
- **M2.1 spike 現場讀的 trl==1.8.0 原始碼**（本機 `pip install --no-deps trl==1.8.0` 裝來
  純讀源碼，不需要 GPU）：`trl/trainer/grpo_trainer.py`（`_generate`/`_prepare_inputs`/
  `_tool_call_loop`/`RepeatSampler` 建構處）、`trl/trainer/utils.py`（`RepeatSampler`、
  `shuffle_sequence_dict`/`permute`）、`trl/trainer/grpo_config.py`（`generation_batch_size`
  預設值推導）、`trl/experimental/openenv/utils.py`（`generate_rollout_completions` 真實實作）
