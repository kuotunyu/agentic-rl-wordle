"""push LoRA adapter 與 merged 模型到 Hugging Face（Colab 收尾 cell 呼叫）。

    python scripts/push_model.py --adapter runs/full/final \
        --repo steven0226/qwen2.5-1.5b-wordle-grpo [--card docs/model_card.md]

- LoRA adapter → <repo>
- merge_and_unload() 後的 bf16 merged → <repo>-merged
- --card 指定的 markdown 會作為兩個 repo 的 README.md（model card）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", type=Path, required=True, help="LoRA adapter 目錄（trainer save_model 輸出）")
    ap.add_argument("--repo", required=True, help="HF repo id，例：steven0226/qwen2.5-1.5b-wordle-grpo")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--merged-repo", default=None, help="預設 <repo>-merged")
    ap.add_argument("--card", type=Path, default=None, help="model card markdown 路徑")
    ap.add_argument("--skip-merged", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    merged_repo = args.merged_repo or f"{args.repo}-merged"

    import torch
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    api = HfApi()

    # ---- LoRA adapter ----
    print(f"[push] LoRA adapter -> {args.repo}", flush=True)
    api.create_repo(args.repo, exist_ok=True, private=args.private)
    api.upload_folder(
        folder_path=str(args.adapter),
        repo_id=args.repo,
        ignore_patterns=["checkpoint-*", "*.log", "optimizer*", "scheduler*", "rng_state*", "trainer_state*"],
    )

    # ---- merged ----
    if not args.skip_merged:
        print(f"[push] 合併權重（bf16）-> {merged_repo}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.base)
        base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, str(args.adapter))
        merged = model.merge_and_unload()
        merged.push_to_hub(merged_repo, private=args.private)
        tokenizer.push_to_hub(merged_repo, private=args.private)

    # ---- model card ----
    if args.card and args.card.exists():
        for repo in ([args.repo] if args.skip_merged else [args.repo, merged_repo]):
            api.upload_file(
                path_or_fileobj=str(args.card),
                path_in_repo="README.md",
                repo_id=repo,
            )
        print(f"[push] model card 已上傳（{args.card}）", flush=True)

    print("[push] 完成 ✅", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
