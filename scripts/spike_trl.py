"""M2.1 選型 spike（60 分鐘 timebox，Colab A100 執行）。

驗證 TRL 1.8 + vLLM colocate + 自訂 rollout_func 能跑出一個乾淨的 optimizer step，
並記錄可用版本三元組。過 → 把版本固定進 requirements-colab.txt、結論寫 docs/decision.md；
不過 → 依 SpikeValidationError 訊息修 rollout.py 的 TRL 接觸面，仍不行改試 ART LocalBackend。

    python scripts/spike_trl.py

檢核點（研究 2026-07-10 的假設逐一驗證）：
  A2  generate_rollout_completions 在 colocate 模式可用、接受 max_tokens/stop 覆寫
  B   rollout_func 收到的 prompts 未依 num_generations 重複（需自回 G 條完成）
  C   rollout_func 額外欄位轉發進 reward_fn 的 kwargs
  D   loss 有限、一步訓練不 crash、checkpoint 可寫
"""

from __future__ import annotations

import sys


def main() -> int:
    import importlib

    print("=== M2.1 spike：版本三元組 ===")
    for mod in ("torch", "vllm", "trl", "transformers", "peft"):
        try:
            m = importlib.import_module(mod)
            print(f"  {mod}=={getattr(m, '__version__', '?')}")
        except ImportError as e:
            print(f"  {mod}: 不可用（{e}）")

    print("\n=== 跑 2 步 smoke preset（number_guess，含 rollout_func 全鏈路）===")
    from wordle_rl.train import main as train_main

    try:
        rc = train_main(
            [
                "--preset",
                "smoke",
                "--max-steps",
                "2",
                "--output-dir",
                "runs/spike",
                "--resume",
                "none",
            ]
        )
    except Exception as e:  # noqa: BLE001 —— spike 就是要把失敗形態印清楚
        import traceback

        traceback.print_exc()
        print(f"\n[SPIKE FAIL] {type(e).__name__}: {e}")
        print(
            "→ 若是 SpikeValidationError：依訊息修 rollout.py 的 _trl_generate_fn / make_rollout_func"
        )
        print("→ 若是 GRPOConfig/GRPOTrainer 旗標錯誤：對照 trl 1.8 docs 修 train.py 的旗標名")
        print("→ 60 分鐘內修不動：轉 ART LocalBackend 備援（docs/decision.md 有指引）")
        return 1

    print("\n[SPIKE PASS] 一步訓練完成。待辦：")
    print("  1. 把上方版本三元組寫進 requirements-colab.txt（精確 pin）")
    print("  2. 把結論與旗標差異記進 docs/decision.md")
    print("  3. 檢查 runs/spike/metrics.jsonl 的 rollout/win_rate 與 reward/mean 非 NaN")
    return rc


if __name__ == "__main__":
    sys.exit(main())
