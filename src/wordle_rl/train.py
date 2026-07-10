"""訓練入口：python -m wordle_rl.train --preset smoke|full [--reward shaped|binary] …

在 Colab（A100，vLLM colocate）或本機 Linux GPU 執行；Windows 本機只負責
開發與測試（本模組的重依賴全部延遲 import）。

【M2.1 spike 驗證點】GRPOConfig / GRPOTrainer 的旗標名以 trl==1.8.0 為準——
研究（2026-07-10）確認 use_vllm / vllm_mode="colocate" / rollout_func 存在，
但 rollout_func 屬實驗性 API；spike 跑通後把可用版本三元組固定進
requirements-colab.txt，如有旗標差異只改本檔與 rollout.py。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from .config import get_preset
from .rollout import RolloutBuffers, make_rollout_func, make_trl_reward_fn


def _fix_missing_cuda13_runtime_ld_path() -> None:
    """已知 Colab A100 環境地雷（M2.1 spike 診斷實錄，第二版修法）：

    vllm 0.23.0 的編譯擴充套件 vllm._C 連到 libcudart.so.13，pip 也確實裝了提供它的
    `nvidia-cuda-runtime`（無 -cu12 尾綴＝cu13）套件，但該 .so 沒有落在動態連結器的
    預設搜尋路徑上。

    第一版修法（只設 os.environ["LD_LIBRARY_PATH"]）**在真實 Colab 環境驗證失敗**：
    LD_LIBRARY_PATH 只在 process 啟動當下被動態連結器讀取一次，process 跑起來後
    用 os.environ 改它對「目前這個 process 自己」後續的 import/dlopen 沒有作用。
    真正有效的做法：用 ctypes.CDLL(..., RTLD_GLOBAL) 把 .so 直接預先載入目前 process
    ——之後任何 import 觸發的 dlopen 找同名庫，連結器會直接命中已載入的這份，
    不需要再查路徑。同時仍然設 LD_LIBRARY_PATH，讓 vllm 自己可能 fork/spawn 出的
    子行程（張量平行/多工作行程，全新 exec 啟動）也能正確讀到路徑。

    找不到 .so 時整個函式是無害 no-op，不影響沒有這個問題的環境
    （例如非 Colab 的本機 Linux GPU）。
    """
    dirs = {
        os.path.dirname(p)
        for p in glob.glob("/usr/**/nvidia/cu13/lib/libcudart.so.13", recursive=True)
    }
    if not dirs:
        return
    os.environ["LD_LIBRARY_PATH"] = (
        ":".join(dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    import ctypes

    for d in dirs:
        so_path = os.path.join(d, "libcudart.so.13")
        if os.path.exists(so_path):
            try:
                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_fix_missing_cuda13_runtime_ld_path()


def _fix_cuda_memory_fragmentation() -> None:
    """已知 Colab 環境地雷（M3.2 pilot 診斷實錄）：Wordle 多輪訓練幾步後穩定 OOM——
    記憶體用量逐步爬升到吃滿整張卡才炸（L4 22GB 第 1 步炸、A100 40GB 第 5 步炸，
    同一種爬升模式只是卡越大撐越多步），符合典型 CUDA allocator 碎片化症狀
    （PyTorch 官方錯誤訊息本身建議這個環境變數）。必須在任何 CUDA 操作發生前設定
    才有效（PyTorch 的 allocator 在首次用到 CUDA 時才讀取這個環境變數初始化，
    跟 LD_LIBRARY_PATH 那種連結器啟動當下才讀一次的情況不同，這裡用 os.environ
    設定是有效的）。
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


_fix_cuda_memory_fragmentation()


def find_resume_checkpoint(output_dir: Path) -> str | None:
    """--resume auto：取 step 最大且 trainer_state.json 完好的 checkpoint；壞則退次新。"""
    candidates = []
    for p in Path(output_dir).glob("checkpoint-*"):
        try:
            step = int(p.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates.append((step, p))
    for _, p in sorted(candidates, reverse=True):
        state = p / "trainer_state.json"
        try:
            json.loads(state.read_text(encoding="utf-8"))
            return str(p)
        except (OSError, json.JSONDecodeError):
            continue
    return None


def build_game_wiring(preset):
    """回傳 (game_factory, answers, initial_messages)。"""
    if preset.game == "wordle":
        from .games import WordleGame
        from .words import get_splits, load_legal

        legal = load_legal()
        answers = list(get_splits(seed=42).train)  # 紅線：eval 詞永不進訓練
        factory = lambda ans: WordleGame(  # noqa: E731
            ans, legal, max_turns=preset.game_max_turns, include_summary=True
        )
        initial = factory(answers[0]).build_messages()  # 開局 messages 與答案無關
    elif preset.game == "number":
        from .env.number_guess import NumberGuessGame

        answers = list(range(1, 101))
        factory = lambda ans: NumberGuessGame(  # noqa: E731
            answer=ans, max_turns=preset.game_max_turns
        )
        initial = factory(1).build_messages()
    else:
        raise ValueError(f"未知 game：{preset.game!r}")
    return factory, answers, initial


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--reward", default="shaped", choices=["shaped", "binary"])
    ap.add_argument("--output-dir", type=Path, default=None, help="預設 runs/<preset>")
    ap.add_argument("--resume", default="auto", help="auto | none | <checkpoint 路徑>")
    ap.add_argument("--max-hours", type=float, default=None, help="牆鐘預算，到點存檔停訓")
    # preset 覆寫（None = 不覆寫）
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--num-generations", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    preset = get_preset(
        args.preset,
        beta=args.beta,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        seed=args.seed,
    )
    output_dir = args.output_dir or Path("runs") / preset.name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] preset={preset.name} game={preset.game} reward={args.reward} "
          f"out={output_dir}", flush=True)

    # ---- 重依賴從這裡開始 ----
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from .callbacks import build_callbacks

    game_factory, answers, initial_messages = build_game_wiring(preset)
    buffers = RolloutBuffers()

    rollout_func = make_rollout_func(
        game_factory=game_factory,
        answers=answers,
        num_generations=preset.num_generations,
        per_turn_max_tokens=preset.per_turn_max_tokens,
        # 送進 rollout 引擎的是「原始生成」預算（不含回饋文字），控制何時停止再生成
        # 下一回合；GRPOConfig 的 max_completion_length（下面）是含回饋文字的整段預算，
        # 兩者不同（M2.1 spike 修正，見 config.py／rollout.py 說明）。
        max_total_tokens=preset.raw_generation_budget,
        temperature=preset.temperature,
        top_p=preset.top_p,
        stop=("</guess>",),
        seed=preset.seed,
        buffers=buffers,
    )
    reward_fn = make_trl_reward_fn(
        game=preset.game,
        preset=args.reward,
        num_generations=preset.num_generations,
        buffers=buffers,
    )

    # dataset row 內容不攜帶答案（答案由 rollout_func 內 seeded sampler 決定，
    # 同組 8 局共用一個）——rows 只是 GRPO 排程的載體，內容全同
    dataset = Dataset.from_list([{"prompt": initial_messages}] * preset.dataset_rows)

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=preset.per_device_batch,
        gradient_accumulation_steps=preset.grad_accum,
        num_generations=preset.num_generations,
        max_completion_length=preset.max_completion_length,
        learning_rate=preset.learning_rate,
        beta=preset.beta,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=preset.warmup_steps,
        temperature=preset.temperature,
        top_p=preset.top_p,
        max_steps=preset.max_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=1_000_000,          # 存檔由 TimedCheckpointCallback 的時間制驅動
        save_total_limit=preset.save_total_limit,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=preset.vllm_gpu_memory_utilization,
        # M3.2 pilot 診斷實錄：OOM 發生在 loss.backward()，此時 vLLM 的權重/KV cache
        # 仍常駐顯存跟訓練側搶空間。開 sleep mode 讓 vLLM 在訓練階段把顯存讓出來，
        # 下一輪生成前才 wake_up() 拿回（trl 1.8 openenv utils 已內建這個機制）。
        vllm_enable_sleep_mode=True,
        report_to=[],
        seed=preset.seed,
    )
    lora_config = LoraConfig(
        r=preset.lora_r,
        lora_alpha=preset.lora_alpha,
        lora_dropout=preset.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=preset.model_name,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        peft_config=lora_config,
        rollout_func=rollout_func,
        callbacks=build_callbacks(
            buffers=buffers,
            samples_dir=output_dir / "samples",
            metrics_path=output_dir / "metrics.jsonl",
            checkpoint_minutes=preset.checkpoint_minutes,
            sample_every_steps=preset.sample_every_steps,
            max_hours=args.max_hours,
        ),
    )

    resume = None
    if args.resume == "auto":
        resume = find_resume_checkpoint(output_dir)
        print(f"[train] resume=auto -> {resume or '全新開跑'}", flush=True)
    elif args.resume not in (None, "none"):
        resume = args.resume

    trainer.train(resume_from_checkpoint=resume)

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    print(f"[train] 完成：LoRA adapter 存於 {final_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
