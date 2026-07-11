# 最終評測報告（真實執行結果）

- 評測集：固定 eval_200（seed=42，n=200），train/eval 嚴格隔離
- 解碼：greedy（do_sample=False），max_new_tokens=160，線索摘要=開
- 產生時間：2026-07-11T15:46:53+00:00

| agent | 勝率 [95% CI] | 勝局均猜 | 非法輸出率 | tag 格式遵循率 | 重用已知不存在字母率 | 破壞已知綠位率 | 重複猜測率 |
|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b-instruct-base | 0.0% [0.0, 1.9] | — | 100.0% | 0.0% | — | — | 0.0% |
| qwen2.5-1.5b **+GRPO LoRA** | 2.0% [0.8, 5.0] | 5.25 | 0.3% | 99.7% | 57.5% | 46.2% | 0.0% |

> 指標口徑同 baselines.md；heuristic 具答案表存取（參照上界），不可與 LLM 直接互比。

## 代表性對局（訓練後模型，含失敗案例）

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
