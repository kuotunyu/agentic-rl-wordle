---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- reinforcement-learning
- grpo
- agentic-rl
- multi-turn
- wordle
- trl
- lora
language:
- en
---

# qwen2.5-1.5b-wordle-grpo

<!-- ⚠️ 模板：斜體佔位段落於 M3.4 評測後以真實數字回填；數據禁止杜撰。 -->

用**多輪 GRPO** 把 `Qwen/Qwen2.5-1.5B-Instruct` 訓練成會玩 Wordle 的 agent：
模型每回合輸出 `<guess>word</guess>`，環境把 G/Y/X 回饋插成 user turn，最多 6 回合。
這是「**訓練 agent 而非 prompt agent**」的示範專案——所有決策能力都寫進權重，
而不是寫進越疊越長的 prompt。

- 訓練框架：TRL 1.8 GRPOTrainer + 自訂多輪 `rollout_func`（單張 A100、vLLM colocate、LoRA）
- 環境/獎勵/評測程式碼：GitHub `agentic-rl-wordle`（含完整 PLAN 與四階段 gate 記錄）
- 姊妹 repo：`-merged` 為合併後全量權重；本 repo 為 LoRA adapter

## 方法

1. **環境**：經典兩趟重複字母規則（先標 G 扣計數，再左→右標 Y）；非法猜測消耗回合。
   答案 2,315 / 合法猜測 12,972（cfreshman 公開清單，fetch-at-setup）。
2. **協定**：system prompt 說明規則與 G/Y/X 記號；歷史每輪
   `You guessed "crane" -> feedback: G Y X X Y` + 已知線索摘要；模型輸出經三態
   robust parser（tag → 全文最後 5 字母詞 → 放棄）。
3. **GRPO**：一組 = 同一隱藏答案的 8 條軌跡（組內比較 advantage）；episode 級獎勵；
   assistant-only loss 以「只有生成 token 進 completion_ids」結構性達成。
4. **訓練詞/評測詞嚴格隔離**：answers 以 seed=42 切 1,852 train / 463 eval，
   評測固定取 eval 前 200 詞，全程不進訓練。

## 獎勵

| 項目 | 值 |
|---|---|
| 獲勝 | +10 + (6−回合數) |
| 新綠位 / 新存在字母（首次發現） | +0.2 / +0.1（shaping 總上界 1.5 << 勝利 10） |
| 非法詞或格式錯誤 | −2 / 回合 |
| 違反已知限制 | −1 / 回合 |
| 重複同一猜測 | −2 / 次 |

量級論證與防 hacking 分析見 repo 的 `docs/rewards.md`（附單元測試）。

## 評測結果（200 個 held-out 詞、greedy、Wilson 95% CI）

*⬜ 佔位：M3.4 跑完 `eval/run_eval.py` 後貼上 random / heuristic / base / +GRPO 四列對照表。*

| agent | 勝率 [95% CI] | 勝局均猜 | 非法輸出率 | tag 遵循率 | 違限率(重用X/破壞G) |
|---|---|---|---|---|---|
| random（12,972 均勻） | 0.0% [0.0, 1.9] | — | 0.0% | 100% | 87.4% / 57.8% |
| heuristic（頻率+過濾；有答案表存取） | 99.5% [97.2, 99.9] | 3.56 | 0.0% | 100% | 0.0% / 0.0% |
| qwen2.5-1.5b-instruct（未訓練） | *⬜* | *⬜* | *⬜* | *⬜* | *⬜* |
| **qwen2.5-1.5b + GRPO LoRA（本模型）** | *⬜* | *⬜* | *⬜* | *⬜* | *⬜* |

## 代表性對局

*⬜ 佔位：5 局 transcript（含至少一局失敗），由 eval/run_eval.py 確定性選取。*

## 限制與誠實聲明

- 1.5B 模型的 Wordle 絕對勝率不高是**預期內**的（HF 官方以 Qwen3-1.7B 跑同任務的結論
  是「有進步但無法穩定獲勝」）。本專案的成功判準是：**勝率顯著超過未訓練 baseline
  （信賴區間佐證）+ 格式錯誤率塌陷 + 違限率下降**，不是「打贏 heuristic solver」。
- heuristic baseline 看得到完整答案分布，是參照上界而非公平對手。
- *⬜ 佔位：訓練後如有未達標項目，在此如實分析原因（獎勵設計/模型容量/訓練量）。*

## 重現

```bash
git clone <repo> && cd agentic-rl-wordle
python scripts/fetch_words.py && pip install -e . && pytest   # 環境全綠
python baselines/run_baseline.py --agent random               # 階段 1
# 訓練：上傳 wordle_rl_bundle.zip 到 Drive，跑 wordle_grpo_colab_train.ipynb（SMOKE→FULL）
python play.py --answer crane --adapter <path>                # 即時觀看
```
