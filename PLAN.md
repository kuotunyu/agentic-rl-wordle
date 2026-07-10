# agentic-rl-wordle — v1 實作計畫

## Context

作品集 R03：用多輪 GRPO 把 `Qwen/Qwen2.5-1.5B-Instruct` 訓練成會玩 Wordle 的 agent。
模型每回合輸出 `<guess>單字</guess>`，環境回傳 G/Y/X 回饋插成 user turn，最多 6 回合，猜中大獎勵。
核心敘事：「別人 prompt agent，我**訓練** agent」。工程嚴謹度是重點——四階段推進，
每階段數字真實落地（gate）才進下一階段。計畫經背景網路查證（2026-07-10，5 路研究）
與雙路設計評審（環境/協定/評測 × 訓練管線/Colab）定案。

**已與使用者確認：**
- HF 使用者名稱 `steven0226` → `steven0226/qwen2.5-1.5b-wordle-grpo`（LoRA）與 `...-merged`
- **所有 GPU 工作（含階段 1 的 Qwen baseline）都上 Colab**；本機（原生 Windows）只做純 CPU 開發與 pytest
- **GitHub 先不發佈**（同專案 2）：git init + 完整 commit 歷史；`gh repo create agentic-rl-wordle --public --source=. --push` + topics 寫進 README 待補
- WSL2 + vLLM 可行（vLLM 官方支援 Linux/WSL2），但只當選配備援，不是依賴路徑

**環境事實：** 資料夾全空、原生 Windows、非 git repo、路徑含中文+全形符號（一出現詭異 build
錯誤就整包搬純 ASCII 路徑，不 debug 路徑）。Colab Pro+（A100 40GB 背景執行）、Colab Secrets
已有 HF_TOKEN、Drive ≥10GB。repo 不上 GitHub → Colab 取碼用 **zip bundle**
（`scripts/make_colab_bundle.py` 打包 → 上傳 Drive → notebook 解壓 + `pip install -e .`，同專案 2 模式）。
本機 Python 3.11 `.venv`；**核心模組（env/protocol/knowledge/rewards/metrics）不 import torch**，
Windows CPU pytest 秒級跑完；torch/transformers 只有 play.py 本機推理才需要。

## 研究結論（2026-07-10 網路查證，來源存 docs/decision.md）

**單字表**（cfreshman 兩個 gist 仍活著且位元組數驗證過；NYT 變體 gist 已 404，勿用）：
- 答案 2,315 詞：`gist.githubusercontent.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b/raw/wordle-answers-alphabetical.txt`
- 額外合法猜測 10,657 詞：`gist.githubusercontent.com/cfreshman/cdcdf777450c5b5301e439061d29694c/raw/wordle-allowed-guesses.txt`
- 合法猜測集 = 兩者聯集 = **12,972**；備援（僅合法集側）：`raw.githubusercontent.com/tabatkins/wordle-list/main/words`（14,855，MIT）
- 慣例是 fetch-at-setup 不入 git（NYT 2024 對 Wordle clone 發過 DMCA，雖非針對詞表，謹慎為上）

**訓練器版圖（推翻規格的 a→b→c 預設順序，理由如下）：**
- **verifiers**：repo 遷至 PrimeIntellect-ai；自帶 GRPOTrainer 於 v0.1.7 棄用，後繼 RLTrainer 拆到
  **未發佈上 PyPI** 的 verifiers-rl；官方訓練路徑改為 prime-rl（三程序、雙 GPU 導向、**無任何 Colab 先例**）
  → 對「單張 Colab A100」實質 blocked。但其 repo 內建 Wordle 環境的 rubric（0.2×新綠+0.1×新黃）是獎勵設計参照。
- **TRL v1.8.0**（2026-07-09）：**官方就有 Wordle 多輪 GRPO 範例**（`examples/scripts/openenv/wordle.py` +
  notebook，Qwen3-1.7B、單 GPU vLLM colocate、num_generations=4、max_completion_length=1024）。
  `rollout_func(prompts, trainer)`（實驗性）在 server 與 colocate 模式都可用；額外回傳欄位會轉發給
  reward functions；assistant-only loss 用「只把模型生成 token 放進 completion_ids」結構性達成
  （Qwen2.5 模板缺 `{% generation %}`，`return_assistant_tokens_mask` 靜默失效——結構性遮罩完全繞開）。
- **ART（openpipe-art 0.5.18）**：API 契合度最高（`TrajectoryGroup` 天生 = 同 scenario 8 條軌跡），
  LocalBackend 單 GPU 可跑（官方 T4 notebook 用 Qwen2.5-3B），但 notebook pin 0.4.11 vs PyPI 0.5.18
  版本撕裂、多個未修 LoRA reload bug（#661/#678/#651）、Drive 續跑要自己黏 → **備援**。
- 兩個校準點：HF 官方發現 Wordle **純輸贏獎勵優於 shaped 獎勵**（做成可切換 preset 對照）；
  Qwen3-1.7B 訓練後「有進步但無法穩定獲勝」→ 1.5B 的成功標準定為「顯著超過未訓練 baseline（CI 佐證）」而非高勝率。

## 定案的關鍵設計決策

| 決策點 | 結論 |
|---|---|
| 訓練器 | **主案 = TRL 1.8 GRPOTrainer + 自訂 rollout_func + in-process 純 Python WordleEnv**（規格 c 路線，被官方 Wordle 範例大幅去風險；不依賴 OpenEnv/WebSocket）。備援 = ART LocalBackend（一個模組可換）。verifiers 僅作環境/獎勵參照。選型評估全文 + 來源 → docs/decision.md |
| 重複字母回饋 | 純函數 `score_guess(guess, answer)` 兩趟：先標 G 並建非綠位置的答案字母 Counter；再左→右對非綠位，計數 >0 標 Y 遞減、否則 X |
| 非法猜測 | **消耗回合**（防無限重試），回饋 = `ILLEGAL` 哨兵（不給字母資訊、不更新線索），user turn 文字「not a valid 5-letter word, this turn is wasted」 |
| 協定語言 | agent 協定（system prompt/回饋訊息）**英文**；文件/README/model card 中文 |
| 歷史渲染 | 過去 assistant turn **正規化**為 `<guess>word</guess>`（原始輸出只進 log）——控制 context 膨脹、去除過期思考；訓練與評測共用同一渲染 |
| 線索摘要 | 預設 **ON**（user turn 附 "Known so far: ..."），config 旗標可關做消融；所有 agent（含 baseline）走同一 build_messages，公平性成立 |
| 切分 | answers 依字母序後以 `random.Random(42)` 洗牌 → train 1,852 / eval 463；**eval_200 = 洗牌尾段前 200**（隨機子集非字母序）；切分即時導出不落地，防漂移 |
| GRPO group | dataset row = 一個 train word；num_generations=8（RepeatSampler 連續重複同 row）→ rollout_func 對每 prompt 自己跑 8 局同答案 episode |
| 答案傳遞 | system prompt 埋 `Game #NNNN` id，train.py 建 `GAME_TABLE[id]→answer`（假設 A1：spike 驗證 dataset 欄位若直達 rollout_func 就簡化掉） |
| KL 係數 | 規格「保守起步」→ FULL 預設 **β=0.01**（PEFT 下 ref logprobs = 關 adapter 再 forward，無額外顯存）；smoke 用 0；`--beta` 旗標留 0 逃生口（若學習太慢）與加大選項（若樣本漂移胡言） |
| 獎勵 | SHAPED（規格版，預設）與 BINARY（HF 發現）雙 preset，`--reward` 切換；訓練先 SHAPED（順帶解 all-zero 組零梯度問題），行有餘力加一晚 BINARY A/B |
| 非法輸出率指標 | parse 三態（tag_ok / fallback_word / no_parse）+ 詞典合法性分開報，非法率 = 非 tag_ok 或環境判非法 的回合占比 |
| 違限指標語義 | (a) 重用已知不存在字母：僅當 knowledge 的 **exact_count==0**（單一 X 於重複字母**不**代表不存在——經典陷阱）；(b) 沒沿用已判 G 位置。分母 = 已有資訊的回合 |
| Colab 取碼 | zip bundle（不依賴 GitHub） |

## 專案佈局

```
（目前資料夾 = repo root）
├── pyproject.toml               # setuptools src layout；package wordle_rl
├── src/wordle_rl/
│   ├── words.py                 # 載入+驗證（斷言 2315/10657/12972）+ get_splits(seed=42)
│   ├── env/wordle.py            # score_guess 純函數 + WordleEnv（reset 必須給 answer；step 後 done 再 step 拋錯）
│   ├── env/number_guess.py      # 煙霧環境：猜 1~100，回饋 higher/lower，≤7 回合
│   ├── protocol.py              # SYSTEM_PROMPT、build_messages、parse_guess（三態）、線索摘要渲染
│   ├── knowledge.py             # Knowledge：greens/min_counts/exact_counts/not_at；update→UpdateDelta（供 shaping）；violations()
│   ├── rewards.py               # 純函數 shaped_reward/binary_reward(EpisodeStats)，零 torch
│   ├── episode.py               # EpisodeStats、transcript 記錄結構
│   ├── agents.py                # RandomAgent / HeuristicAgent / LLMAgent（全部輸出含 <guess> 的 raw text，走同一 parser）
│   ├── backends.py              # GenerationBackend 協定：TransformersBackend / VLLMBackend
│   ├── runner.py                # 推理側 lockstep 批次對局 runner（baseline 與 eval 共用，公平性關鍵）
│   ├── metrics.py               # wilson_ci、勝率/非法率/違限率彙整
│   ├── rollout.py               # ★唯一碰 TRL 的模組：make_rollout_func、reward-fn 轉接、樣本 ring buffer
│   ├── callbacks.py             # TimedCheckpoint(30min)、SampleDump(50步3局)、MetricsJsonl、--max-hours 優雅收尾
│   ├── train.py                 # __main__：--preset smoke|full --reward shaped|binary --resume auto --max-hours N
│   └── config.py                # SMOKE / FULL dataclass preset
├── scripts/fetch_words.py       # stdlib urllib、原子寫入、計數斷言、SOURCE.json 記錄 URL+sha256、--fallback 僅換合法集側
├── scripts/make_colab_bundle.py # 原始碼 zip（排除 data/.venv/checkpoints）
├── scripts/push_model.py        # LoRA adapter push + merge_and_unload() bf16 merged push + model card
├── baselines/run_baseline.py    # 薄 CLI over runner：--agent random|heuristic|llm --backend transformers|vllm
├── eval/run_eval.py             # 薄 CLI over runner：前後對照、Wilson CI、對照表、5 局 transcript
├── play.py                      # --answer WORD [--adapter DIR] [--device auto]；逐回合印原始輸出/解析/彩色回饋（colorama）
├── tests/                       # test_env(14+案)/test_knowledge/test_protocol/test_rewards/test_words/test_number_guess
├── docs/decision.md             # 選型結論與研究來源
├── docs/rewards.md              # 獎勵量級理由與防 hacking 分析
├── wordle_grpo_colab_train.ipynb
├── data/（gitignored+.gitkeep）、results/、samples/
└── PLAN.md、README.md、LICENSE(Apache-2.0)、.gitignore、requirements.txt、requirements-colab.txt
```

## 階段 0：環境與協定（本機 CPU，gate：pytest 全綠）

**回饋演算法預驗證測試表**（已逐案人工用兩趟演算法追蹤核對，直接進 tests/test_env.py）：

| # | answer | guess | 預期 | 驗證點 |
|---|---|---|---|---|
| 1 | apple | paper | YYGYX | 交錯雙 p：G 吃掉一個，剩一個 Y |
| 2 | abide | speed | XXYXY | 猜雙 e 答單 e：左先 Y、右 X（左到右封頂） |
| 3 | those | geese | XXXGG | G 先吃計數：唯一 e 被綠位吃掉，前面 e 全 X |
| 4 | geese | eagle | YXYXG | 答三 e：綠吃一個，開頭 e 仍 Y |
| 5 | abbey | babes | YYGGX | 雙 b 一對齊：G+Y |
| 6 | abbey | kebab | XYGYY | 猜雙 b 答雙 b：G+尾 Y |
| 7 | banal | mamma | XGXXY | 三 m 全 X；雙 a → 一 G 一 Y |
| 8 | mamma | mummy | GXGGX | 答三 m 全綠化，其餘 X |
| 9 | sleep | eerie | YYXXX | 猜三 e 答雙 e：恰兩 Y 第三 X |
| 10 | hello | llama | YYXXX | 雙 l 對雙 l 無對齊：兩 Y 封頂 |
| 11 | hello | label | YXXYY | 雙 l 拆位：皆 Y |
| 12 | dolly | lolly | XGGGG | 綠位吃光兩個 l → 開頭 l 是 X 不是 Y |
| 13 | silly | lolly | XXGGG | 同上、兩個前導 X |
| 14 | crane | crane | GGGGG | 勝利；env 斷言 done/won |

另測：非法詞消耗回合回 ILLEGAL、None guess、6 次非法 = 敗局且 info 揭答案、大小寫/空白正規化、done 後 step 拋錯。

**protocol.py**：system prompt 含規則、G/Y/X 記號說明（含重複字母語義一句話）、
「至多兩句簡短思考後輸出唯一一個 `<guess>word</guess>`」（短 completion = 便宜 rollout）。
parser 順序：最後一個 `<guess>([a-zA-Z]+)</guess>`（5 字母→tag_ok）→ 全文最後一個獨立 5 字母詞
（fallback_word）→ no_parse（guess=None）。

**knowledge.py** update 規則（每 guess 每字母 L，g/y/x = 本次 G/Y/X 數）：
`min_counts[L]=max(old, g+y)`；**x>0 ⇒ exact_counts[L]=g+y**（涵蓋「素 X → 0」與「重複字母第二份 X → 精確計數」）；
G 記 greens[i]；非 G 出現位記 not_at[i]。ILLEGAL 回合不 update。`is_consistent()` 供啟發式過濾，
`UpdateDelta`（新綠數/計數增量）供獎勵 shaping。

## 階段 1：baseline（gate：results/baselines.md 三行真實數字）

- **RandomAgent**：對 12,972 合法集均勻抽（seeded）——真地板。
- **HeuristicAgent**：候選 = 2,315 答案表過濾 `is_consistent`；分數 = 逐位字母頻率和（對當前候選集計算），
  重複字母第二次起減半權重；猜 argmax（永遠猜可能答案）；平手取字典序最小（全確定性）。
  **文件明載 leakage**：啟發式看得到完整答案分布（Wordle solver 常規，作為參照上界）。
- **LLMAgent**（Colab vLLM）：runner 以 lockstep 批次跑 200 局——每回合對 active episodes 發**一次**批次生成
  （≤6 次批次呼叫跑完全部）；vLLM `SamplingParams(temperature=0)`；Qwen generation_config 預設帶採樣，
  **必須顯式關掉**（transformers 側 do_sample=False 並清 temperature/top_p/top_k）。
- 指標：勝率 + Wilson 95% CI、勝局平均猜測數（含浪費回合）、非法輸出率（tag/詞典兩層分列）、
  重用已知不存在字母率、破壞已知 G 率（分母 = 已有資訊回合）。
- 產出 `results/baselines.json`（逐局原始記錄）+ `results/baselines.md`（表）。
  random/heuristic 本機 CPU 直接跑；LLM baseline 在 Colab 用同一 CLI `--backend vllm` 跑。

## 階段 2：選型 spike + 煙霧測試（gate：煙霧環境學習曲線真實上升）

- **M2.1 Spike（60 分鐘，Colab A100）**：trl==1.8.0 + vLLM colocate + 極簡 rollout_func 對 10 個 dummy prompt
  跑一個乾淨 optimizer step。驗證：prompt slice 語義（不重複、需自回 8 條）、
  `trl.experimental.openenv.generate_rollout_completions` 在 colocate 可用且接受停止字串/max_tokens 覆寫（假設 A2，
  不行就直呼 trainer 的 vLLM engine）、額外欄位轉發 reward fn、dataset 欄位是否直達（A1）。
  **記錄可用的 (torch, vllm, trl) 版本三元組進 requirements-colab.txt**——這是 Colab 重現性的錨。
  結論 + 理由 + 研究來源寫 docs/decision.md。若 spike 失敗超時 → 改試 ART LocalBackend（tic_tac_toe notebook 為模板）。
- **煙霧環境**：number_guess（1~100、higher/lower、≤7 回合、同 `<guess>` 協定、同 rollout 機制）。
  獎勵：勝 +5+0.5×(7−turns)、格式錯 −2、重複 −2（回饋已密集，不需資訊 shaping）。dataset 512 rows。
- **煙霧 gate（SMOKE preset，A100 ~20–30 分鐘、40–60 步）**：零 crash；`len(logprobs)==len(completion_ids)` 斷言成立；
  尾 20 步平均 reward > 前 10 步且斜率可見；格式錯誤率 <5%；勝率較未訓練 +15 個百分點以上。
- **M2.4 續跑演練（強制里程碑）**：中途 kill，`--resume auto` 接續，global_step 連續、曲線無斷崖。

## 階段 3：Wordle 正式訓練 + 評測

**獎勵（rewards.py 純函數 + pytest + docs/rewards.md）**

| 項目 | 值 | 界限/理由 |
|---|---|---|
| 獲勝 | +10 + (6−turns_used) | 10~15；快勝加成嚴格遞減，堵「拖時間」 |
| 新 G（首次發現，含 Y→G 升級） | +0.2 | ≤1.0 |
| 新 Y（首次發現） | +0.1 | ≤0.5；shaping 總上界 1.5 << 最小勝利獎勵 10 → 「刷 shaping 不求勝」被 ≥8.5 差距支配 |
| 非法詞/格式錯 | −2/回合 | 大於單回合最大 shaping +0.7 |
| 違反已知限制 | −1/回合 | 溫和為之：探測性猜詞是合法策略，只輕罰 |
| 重複同一猜測 | −2/次 | 防 hacking：重複零資訊且必虧 |

episode 級單一純量（各項相加）= GRPO 一條軌跡一個 advantage 的正確形態。
BINARY preset：勝 1 / 敗 0。預設 SHAPED（shaping 造成組內變異，緩解早期全敗零梯度），
`group_zero_variance_frac` 進 metrics.jsonl 監控；`scale_rewards="none"`（Dr.GRPO）留逃生口。

**訓練架構（rollout.py，唯一 TRL 接觸面）**：每 prompt 開 8 局同答案 episode；
**回合 lockstep**——所有 active episode 湊一批呼叫一次生成（stop=`</guess>` 且含停止字串於輸出，
否則解析文字與 loss 視窗漂移）；每局把各回合生成 token/logprobs 串接（vLLM 取樣時 logprobs，
importance-sampling 近似，與官方範例同）；回饋只進 prompt 側 → 結構性 assistant-only loss；
單局 token 預算 6×160=960 ≤ max_completion_length 1024，爆預算記敗局。
解析失敗 → env.step(None) 消耗回合 + 懲罰 + 「Invalid format...」回饋，局不中斷。
rollout 溫度 1.0（組內變異是 GRPO 命脈）；評測另走 greedy 路徑，永不用 trainer 生成。

**超參（config.py preset）**

| 參數 | SMOKE（猜數字） | FULL（Wordle，A100 40GB） |
|---|---|---|
| LoRA r/α/dropout | 16/32/0.05 | 同左；target = q,k,v,o,gate,up,down_proj |
| lr / schedule | 1e-5 constant + 10 warmup | 同左 |
| β (KL) | 0.0 | **0.01**（PEFT 關 adapter 取 ref logprobs，無額外顯存；`--beta` 可調） |
| num_generations | 8 | 8（同答案一組） |
| batch×accum | 8×1（1 組/步） | 8×2（2 組=16 局/步） |
| max_prompt / completion(episode 總) | 512 / 512 | 512 / 1024 |
| 每回合 max_tokens / stop | 64 / `</guess>` | 160 / `</guess>` |
| vLLM colocate | util 0.25 + sleep_mode | 同左（1.5B 權重 ~3GB，OOM 時 util→0.2、batch 4×4、回合 128 tok） |
| 精度/顯存 | bf16 + grad ckpt | 同左 |
| 步數/牆鐘 | 60 步 / 0.5h | 400 步 / `--max-hours 8`（估 60–120s/步，A3 於 M3.2 實測校正） |
| checkpoint | 10 分鐘 | **TimedCheckpointCallback 30 分鐘**（on_step_end 看 monotonic 設 should_save，免猜 sec/step）；save_total_limit=3；直接寫 Drive |
| 樣本/指標 | 25 步 | **50 步 3 局 transcript → samples/step_N.md**；metrics.jsonl（reward、win_rate、format_err、repeat_rate、zero-variance 組占比） |

dataset = 1,852 train words（eval 463 永不進）；400 步 × 2 組 ≈ 800 個答案輪替。

**wordle_grpo_colab_train.ipynb**（沿用專案 1/2 慣例）：①參數 cell（SMOKE_TEST、RUN_NAME、HF_USERNAME=steven0226、
MAX_HOURS、REWARD_PRESET）→ ②解壓 Drive 的 bundle + 精確 pin 安裝（斷言 trl.__version__）→
③掛 Drive、CKPT_DIR=Drive/runs/{RUN_NAME} → ④userdata.get("HF_TOKEN") → ⑤`pytest -q`（<60s，抓環境漂移）→
⑥`subprocess.Popen([sys.executable,"-m","wordle_rl.train",...,"--resume","auto","--max-hours",...])`，
log 落 Drive（cell 直跑會被 websocket 拖死）→ ⑦監控 cell：tail log + 畫 metrics.jsonl 曲線 →
⑧收尾：run_eval 前後對照 → push_model.py（LoRA + merged）→ `finally: runtime.unassign()`（push 失敗不擋釋放，
Drive 是真相來源可事後補 push）。TRL checkpoint 含 optimizer/RNG/trainer_state，取樣是 (seed, global_step)
純函數 → resume 正確性由構造保證；`--resume auto` = 取最大 step 有效 checkpoint，壞則退次新。

**評測（eval/run_eval.py，gate：紅線標準）**：同 200 eval 詞、同協定同摘要設定（不合 = 硬錯）、
greedy、base 與 base+adapter 同 seed 兩趟；表 = random / heuristic / base LLM / RL LLM ×
勝率[Wilson 95% CI] / 勝局均猜 / 非法率 / 兩violate率 → `results/final_report.md`；
transcript 確定性選取：≤3 回合勝 ×2、5–6 回合勝 ×2、敗局 ×1 → README + model card。

## 交付
- HF：`steven0226/qwen2.5-1.5b-wordle-grpo`（LoRA）+ `-merged`；model card 中文：方法、獎勵表、
  對照表（含 CI）、5 局 transcript、限制與失敗案例誠實呈現、詞表出處、Apache-2.0。
- GitHub 暫緩（發佈指令寫 README 待補）；本機 git 完整歷史。
- play.py、README：mermaid 架構圖（協定→rollout→GRPO→eval 閉環）、四階段流程、成果表、
  「prompt agent vs 訓練 agent」論述一段、詞表來源與抓取說明。

## 實作順序（里程碑，每個 gate 擋下一步）

1. **M0a 骨架**：.venv(3.11) + pyproject + git init + .gitignore；fetch_words 抓表驗數；PLAN.md 入 repo。
2. **M0b 環境全綠**：env/protocol/knowledge/words + 全部測試（含 14 案回饋表）Windows CPU pytest 全綠。
3. **M1 baseline**：本機跑 random/heuristic；Colab 跑 LLM baseline（vLLM）→ results/baselines.md。
4. **M2.1 spike**（60 分鐘 timebox）→ docs/decision.md + 版本三元組。
5. **M2.2 訓練管線本機建置**：number_guess + rewards + rollout + train + callbacks，pytest 全綠（CPU）。
6. **M2.3 煙霧訓練**（A100 20–30 分）達 gate；**M2.4 續跑演練**。
7. **M3.1 接上 Wordle**（環境已綠，只換 env_factory + 獎勵 preset）；**M3.2 試跑 1 小時**：
   50 步內格式錯 <10%、zero-variance 組 <40%、transcript 目視無 hacking；實測 sec/step 校正步數/checkpoint 數學。
8. **M3.3 過夜 FULL（SHAPED，~8h）**；醒來看曲線與 samples/，視情況第二晚（BINARY A/B 或續訓）。
9. **M3.4 評測**：run_eval 出 final_report.md（CI 佐證）。
10. **M3.5 收尾**：push HF（LoRA+merged+model card）、README、git commit 完整歷史。

## 驗證方式
- 每階段 gate 數字全部來自真實執行：pytest 輸出、baselines.md、煙霧曲線、metrics.jsonl、final_report.md。
- 續跑演練實測（kill → resume → step 連續）。
- 訓練期 samples/ 每 50 步 transcript 人工抽查 reward hacking（重複猜、違限刷分、拖延）。
- play.py 本機實際解一個 eval 詞並回報過程。

## 風險與備案

| 風險 | 等級 | 備案 |
|---|---|---|
| rollout_func 實驗性 API 變動 | 中 | pin trl==1.8.0；TRL 面全部關進 rollout.py+callbacks.py；換 ART = 換一個模組 |
| 早期全敗零梯度組 | 高（BINARY）| 預設 SHAPED；監控 zero-variance 占比；G=8；scale_rewards="none" 逃生口 |
| Colab pin 衝突（torch/vllm/trl） | 中 | spike 記錄可用三元組；安裝 cell 斷言版本；README 留已知可用組合 |
| Colab 半夜斷線 | 高 | 30 分 Drive checkpoint + --resume auto + 續跑演練是強制里程碑；睡前起床各看一眼 session |
| vLLM colocate OOM | 中 | util 0.25+sleep_mode+grad ckpt+β 可歸零；再不行 batch 4×4、回合 128 tok |
| 1.5B 勝率絕對值低 | 高 | 成功標準 = 顯著超過未訓練 baseline（CI）+ 格式錯誤率塌陷 + reward 曲線；README 從第一天就這樣框定；真不行則如實寫限制+原因分析（合格作品）。備選槓桿（v1 外）：few-shot 協定、SFT 暖身 |
| 中文路徑詭異錯誤 | 低 | 立即整包搬純 ASCII 路徑 |
| R03 卡關 >2 天 | — | 依手冊凍結、啟用 R-BENCH 替補 |

## 紅線
- 每階段數據真實落地才進下一階段；勝率沒顯著超過未訓練 baseline 就不准宣稱成功，如實寫限制與分析。
- 訓練詞（1,852）與評測詞（463/200）嚴格隔離；所有隨機固定 seed 可重現。
- 曲線不可信、樣本才可信：transcript 抽查是例行公事，不是出事才做。
