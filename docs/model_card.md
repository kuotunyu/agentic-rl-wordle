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

## 評測結果（200 個 held-out 詞、greedy、Wilson 95% CI；2026-07-11 真實執行）

| agent | 勝率 [95% CI] | 勝局均猜 | 非法輸出率 | tag 遵循率 | 違限率(重用X/破壞G) |
|---|---|---|---|---|---|
| random（12,972 均勻） | 0.0% [0.0, 1.9] | — | 0.0% | 100% | 87.4% / 57.8% |
| heuristic（頻率+過濾；有答案表存取） | 99.5% [97.2, 99.9] | 3.56 | 0.0% | 100% | 0.0% / 0.0% |
| qwen2.5-1.5b-instruct（未訓練） | 0.0% [0.0, 1.9] | — | **100.0%** | **0.0%** | — |
| **qwen2.5-1.5b + GRPO LoRA（本模型）** | **2.0% [0.8, 5.0]** | 5.25 | **0.3%** | **99.7%** | 57.5% / 46.2% |

**判讀（照本專案自己立的紅線誠實陳述）：**

- **格式與合法性：塌陷式學會**。未訓練的 base 在此協定下 100% 回合非法（連一手合法棋
  都下不出來）；訓練後 99.7% 回合輸出格式正確的合法 5 字母單字。這是壓倒性顯著的差異，
  也是本次訓練最確定的成果。
- **勝率：方向正確、未達統計顯著**。0/200 → 4/200（0.0% → 2.0%），但 Fisher 精確檢定
  單尾 p≈0.06、兩者 Wilson CI 有重疊——以 n=200 不足以宣稱「勝率顯著超過 baseline」。
  依紅線，本 model card 不做此宣稱。
- **無 reward hacking**：重複猜測率 0.0%，訓練期 samples/ 人工抽查亦無刷分模式。

## 代表性對局（訓練後模型，eval 確定性選取，含失敗案例）

### answer: `shiny` — WIN in 5 turn(s)

- turn 1: guess=`cloud` [tag_ok] -> `XXXXX`
- turn 2: guess=`train` [tag_ok] -> `XXXYY`
- turn 3: guess=`spike` [tag_ok] -> `GXGXX`
- turn 4: guess=`snake` [tag_ok] -> `GYXXX`  ⚠ breaks_green, reuses_absent_letter
- turn 5: guess=`shiny` [tag_ok] -> `GGGGG`

### answer: `shame` — WIN in 6 turn(s)

- turn 1: guess=`cloud` [tag_ok] -> `XXXXX`
- turn 2: guess=`train` [tag_ok] -> `XXGXX`
- turn 3: guess=`brave` [tag_ok] -> `XXGXG`  ⚠ reuses_absent_letter
- turn 4: guess=`spare` [tag_ok] -> `GXGXG`  ⚠ reuses_absent_letter
- turn 5: guess=`smash` [tag_ok] -> `GYGXY`  ⚠ breaks_green
- turn 6: guess=`shame` [tag_ok] -> `GGGGG`

### answer: `large` — LOSS in 6 turn(s)

- turn 1: guess=`cloud` [tag_ok] -> `XYXXX`
- turn 2: guess=`train` [tag_ok] -> `XYYXX`
- turn 3: guess=`brave` [tag_ok] -> `XYYXG`
- turn 4: guess=`pearl` [tag_ok] -> `XYYYY`  ⚠ breaks_green
- turn 5: guess=`glory` [tag_ok] -> `YYXYX`  ⚠ breaks_green, reuses_absent_letter
- turn 6: guess=`smear` [tag_ok] -> `XXYYY`  ⚠ breaks_green

### answer: `pedal` — LOSS in 6 turn(s)

- turn 1: guess=`cloud` [tag_ok] -> `XYXXY`
- turn 2: guess=`flank` [tag_ok] -> `XYYXX`
- turn 3: guess=`blame` [tag_ok] -> `XYYXY`
- turn 4: guess=`glade` [tag_ok] -> `XYYYY`
- turn 5: guess=`train` [tag_ok] -> `XXYXX`  ⚠ reuses_absent_letter
- turn 6: guess=`plead` [tag_ok] -> `GYYGY`

### answer: `frown` — LOSS in 6 turn(s)

- turn 1: guess=`cloud` [tag_ok] -> `XXGXX`
- turn 2: guess=`stone` [tag_ok] -> `XXGYX`
- turn 3: guess=`brain` [tag_ok] -> `XGXXG`  ⚠ breaks_green
- turn 4: guess=`green` [tag_ok] -> `XGXXG`  ⚠ breaks_green, reuses_absent_letter
- turn 5: guess=`crane` [tag_ok] -> `XGXYX`  ⚠ breaks_green, reuses_absent_letter
- turn 6: guess=`flank` [tag_ok] -> `GXXYX`  ⚠ breaks_green, reuses_absent_letter

## 限制與誠實聲明

- 1.5B 模型的 Wordle 絕對勝率不高是**預期內**的（HF 官方以 Qwen3-1.7B 跑同任務的結論
  是「有進步但無法穩定獲勝」）。本專案的成功判準是：**勝率顯著超過未訓練 baseline
  （信賴區間佐證）+ 格式錯誤率塌陷 + 違限率下降**，不是「打贏 heuristic solver」。
- heuristic baseline 看得到完整答案分布，是參照上界而非公平對手。
- **勝率項未達統計顯著**（0/200 → 4/200，Fisher 單尾 p≈0.06）：三項判準中，格式錯誤率
  塌陷（100%→0.3%）與訓練期 reward 曲線持續上升（-9.4 → -3.2，3000 步）確定成立，
  勝率項方向正確但差距不足以在 n=200 下過檢定。如實記錄，不宣稱勝率顯著。
- **策略弱點（訓練期與評測一致觀察到）**：模型收斂到固定開局腳本（cloud → train →…），
  且經常不沿用已確認的綠位（破壞綠位率 46.2%）、重用已排除字母（57.5%）——它學會了
  「下合法的棋」，但只部分學會「利用回饋收斂」。可能原因依序：(a) shaped 獎勵中
  「非法 −2/回合」是最大的梯度訊號，先被吃掉後策略梯度變稀疏；(b) 1.5B 容量對
  多步約束推理吃緊；(c) 3000 步（4.8 萬局）對策略層次的學習仍偏短。
- **可能的後續槓桿（v1 未做）**：binary 獎勵 A/B（HF 官方發現對 Wordle 更穩）、
  提高違限懲罰或對「利用新資訊」加大 shaping、SFT 暖身、更大 base 模型。

## 重現

```bash
git clone <repo> && cd agentic-rl-wordle
python scripts/fetch_words.py && pip install -e . && pytest   # 環境全綠
python baselines/run_baseline.py --agent random               # 階段 1
# 訓練：上傳 wordle_rl_bundle.zip 到 Drive，跑 wordle_grpo_colab_train.ipynb（SMOKE→FULL）
python play.py --answer crane --adapter <path>                # 即時觀看
```
