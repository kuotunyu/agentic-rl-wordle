# 最終完整評測報告（真實執行結果）

- 評測集：固定 eval_full（seed=42，n=463），train/eval 嚴格隔離
- 解碼：greedy（do_sample=False），max_new_tokens=160，線索摘要=開
- 產生時間：2026-07-26T16:00:32+00:00
- 原始 Colab 報告的 base label 是本機 HF snapshot hash；此處正規化為公開模型名稱，數值未修改

| agent | 勝率 [95% CI] | 勝局均猜 | 非法輸出率 | tag 格式遵循率 | 重用已知不存在字母率 | 破壞已知綠位率 | 重複猜測率 |
|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b-instruct-base | 0.0% [0.0, 0.8] | — | 100.0% | 0.0% | — | — | 0.0% |
| qwen2.5-1.5b **+GRPO LoRA** | 2.8% [1.6, 4.7] | 4.08 | 0.2% | 99.9% | 58.5% | 48.9% | 0.0% |

> 指標口徑同 baselines.md；heuristic 具答案表存取（參照上界），不可與 LLM 直接互比。

## 代表性對局（訓練後模型，含失敗案例）

### answer: `stone` — WIN in 2 turn(s)

- turn 1: guess=`cloud` [tag_ok] -> `XXGXX`
- turn 2: guess=`stone` [tag_ok] -> `GGGGG`

### answer: `brute` — WIN in 3 turn(s)

- turn 1: guess=`cloud` [tag_ok] -> `XXXYX`
- turn 2: guess=`train` [tag_ok] -> `YGXXX`
- turn 3: guess=`brute` [tag_ok] -> `GGGGG`

### answer: `shiny` — WIN in 5 turn(s)

- turn 1: guess=`cloud` [tag_ok] -> `XXXXX`
- turn 2: guess=`train` [tag_ok] -> `XXXYY`
- turn 3: guess=`spike` [tag_ok] -> `GXGXX`
- turn 4: guess=`snake` [tag_ok] -> `GYXXX`  ⚠ breaks_green, reuses_absent_letter
- turn 5: guess=`shiny` [tag_ok] -> `GGGGG`

### answer: `crack` — WIN in 4 turn(s)

- turn 1: guess=`cloud` [tag_ok] -> `GXXXX`
- turn 2: guess=`chain` [tag_ok] -> `GXGXX`
- turn 3: guess=`crate` [tag_ok] -> `GGGXX`
- turn 4: guess=`crack` [tag_ok] -> `GGGGG`

### answer: `large` — LOSS in 6 turn(s)

- turn 1: guess=`cloud` [tag_ok] -> `XYXXX`
- turn 2: guess=`train` [tag_ok] -> `XYYXX`
- turn 3: guess=`brave` [tag_ok] -> `XYYXG`
- turn 4: guess=`pearl` [tag_ok] -> `XYYYY`  ⚠ breaks_green
- turn 5: guess=`glory` [tag_ok] -> `YYXYX`  ⚠ breaks_green, reuses_absent_letter
- turn 6: guess=`smear` [tag_ok] -> `XXYYY`  ⚠ breaks_green
