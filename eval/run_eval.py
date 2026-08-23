"""階段 3 評測：訓練前後模型在同一批 held-out 詞（固定 seed、greedy）對照。

    python eval/run_eval.py --adapter runs/full/final --backend vllm          # Colab
    python eval/run_eval.py --adapter <HF_REPO> --backend vllm \
        --split eval_full --n 463 --out results/full_463_report.md
    python eval/run_eval.py --adapter runs/full/final --backend transformers  # 本機 GPU

- base 與 base+LoRA 用完全相同的協定/詞序/greedy 設定各跑一遍。
- 預設 200 詞評測會與 results/baselines.json 的 random / heuristic / base 列合併成
  results/final_report.md 對照表（勝率附 Wilson 95% CI）。
- ``--split eval_full --n 463`` 評估完整 held-out 集；不混入只有 200 詞的舊 baseline。
- 從訓練後模型的對局中確定性挑 5 局代表 transcript（含至少一局失敗）
  → results/transcripts/ 並內嵌到報告。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from wordle_rl.agents import LLMAgent
from wordle_rl.backends import GenConfig
from wordle_rl.episode import EpisodeStats
from wordle_rl.metrics import AggregateMetrics, aggregate
from wordle_rl.runner import run_episodes
from wordle_rl.words import get_splits, load_legal

REPO_ROOT = Path(__file__).resolve().parents[1]

METRIC_COLUMNS = [
    ("win_rate", "勝率 [95% CI]"),
    ("avg_guesses_on_wins", "勝局均猜"),
    ("illegal_rate", "非法輸出率"),
    ("tag_ok_rate", "tag 格式遵循率"),
    ("absent_reuse_rate", "重用已知不存在字母率"),
    ("green_break_rate", "破壞已知綠位率"),
    ("repeat_rate", "重複猜測率"),
]


def make_backend(args, adapter: str | None):
    if args.backend == "vllm":
        from wordle_rl.backends import VLLMBackend

        return VLLMBackend(args.model, adapter=adapter)
    from wordle_rl.backends import TransformersBackend

    return TransformersBackend(args.model, adapter=adapter)


def run_llm_with_backend(args, backend, words, legal) -> list[EpisodeStats]:
    cfg = GenConfig(max_new_tokens=args.max_new_tokens, do_sample=False, stop=("</guess>",))
    agent = LLMAgent(backend, cfg, include_summary=not args.no_summary)
    return run_episodes(agent, words, legal)


def run_llm(args, adapter: str | None, words, legal) -> list[EpisodeStats]:
    backend = make_backend(args, adapter)
    try:
        return run_llm_with_backend(args, backend, words, legal)
    finally:
        del backend


def select_eval_words(splits, split: str, n: int) -> list[str]:
    """Select exactly ``n`` words from the requested deterministic split."""
    pool = getattr(splits, split)
    if n <= 0:
        raise ValueError("--n must be positive")
    if n > len(pool):
        raise ValueError(
            f"--n={n} exceeds {split} size ({len(pool)}); "
            f"use --split eval_full for all {len(splits.eval_full)} held-out words"
        )
    return list(pool[:n])


def pick_transcripts(episodes: list[EpisodeStats], k: int = 5) -> list[EpisodeStats]:
    """確定性挑選：≤3 回合勝 ×2 → 4–6 回合勝 ×2 → 敗局 ×1；缺類依序遞補。"""
    fast_wins = [e for e in episodes if e.won and e.turns_used <= 3]
    slow_wins = [e for e in episodes if e.won and e.turns_used > 3]
    losses = [e for e in episodes if not e.won]
    picked: list[EpisodeStats] = fast_wins[:2] + slow_wins[:2] + losses[:1]
    for pool in (losses, slow_wins, fast_wins, episodes):
        for e in pool:
            if len(picked) >= k:
                break
            if e not in picked:
                picked.append(e)
    return picked[:k]


def metrics_payload(m: AggregateMetrics) -> dict:
    return {
        "n": m.n_episodes,
        "wins": m.wins,
        "win_rate": m.win_rate,
        "win_ci": [m.win_ci_low, m.win_ci_high],
        "avg_guesses_on_wins": m.avg_guesses_on_wins,
        "illegal_rate": m.illegal_rate,
        "env_illegal_rate": m.env_illegal_rate,
        "tag_ok_rate": m.tag_ok_rate,
        "absent_reuse_rate": m.absent_reuse_rate,
        "green_break_rate": m.green_break_rate,
        "repeat_rate": m.repeat_rate,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter 路徑（訓練後模型）")
    ap.add_argument("--backend", default="transformers", choices=["transformers", "vllm"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument(
        "--split",
        default="eval_200",
        choices=["eval_200", "eval_full"],
        help="eval_200（預設可對照既有 baseline）或完整 463 詞 eval_full",
    )
    ap.add_argument("--seed", type=int, default=42, help="切分 seed（勿改，紅線）")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument(
        "--skip-base",
        action="store_true",
        help="沿用 results/baselines.json 既有的 base 模型列（相同協定時省一輪推理）",
    )
    ap.add_argument(
        "--label-base",
        default=None,
        help="報告中的 base 名稱；model 是 HF snapshot 本機路徑時建議明確指定",
    )
    ap.add_argument("--label-tuned", default="qwen2.5-1.5b **+GRPO LoRA**")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "final_report.md")
    ap.add_argument("--baselines", type=Path, default=REPO_ROOT / "results" / "baselines.json")
    ap.add_argument("--transcripts-dir", type=Path, default=REPO_ROOT / "results" / "transcripts")
    args = ap.parse_args()

    splits = get_splits(seed=args.seed)
    try:
        words = select_eval_words(splits, args.split, args.n)
    except ValueError as exc:
        ap.error(str(exc))
    legal = load_legal()
    can_reuse_baselines = args.split == "eval_200" and len(words) == 200
    if args.skip_base and not can_reuse_baselines:
        ap.error("--skip-base is only valid for the existing eval_200, n=200 baseline")

    rows: list[tuple[str, dict]] = []  # (label, metrics_row) 依表序
    payloads: dict[str, dict] = {}

    # ---- baseline 列（random / heuristic）----
    base_label = args.label_base or args.model.split("/")[-1].lower() + "-base"
    if can_reuse_baselines and args.baselines.exists():
        store = json.loads(args.baselines.read_text(encoding="utf-8"))
        for name in ("random", "heuristic"):
            if name in store.get("agents", {}):
                rows.append((name, store["agents"][name]["metrics_row"]))
                payloads[name] = store["agents"][name]["metrics"]

    # ---- base 模型 ----
    base_row_from_store = None
    if args.skip_base and args.baselines.exists():
        store = json.loads(args.baselines.read_text(encoding="utf-8"))
        base_row_from_store = store.get("agents", {}).get(base_label)
    shared_vllm_backend = None
    if base_row_from_store is not None:
        print(f"[eval] base 沿用 baselines.json 的 {base_label} 列", flush=True)
        rows.append((base_label, base_row_from_store["metrics_row"]))
        payloads[base_label] = base_row_from_store["metrics"]
    else:
        print(f"[eval] 跑 base 模型（{args.backend}, greedy, n={len(words)}）…", flush=True)
        if args.backend == "vllm" and args.adapter:
            # Load one engine with LoRA support, then evaluate base and adapter
            # through the same weights. This avoids re-initializing CUDA/vLLM
            # and cuts full-463 setup time roughly in half.
            shared_vllm_backend = make_backend(args, adapter=args.adapter)
            shared_vllm_backend.adapter = None
            base_eps = run_llm_with_backend(args, shared_vllm_backend, words=words, legal=legal)
        else:
            base_eps = run_llm(args, adapter=None, words=words, legal=legal)
        m = aggregate(base_eps)
        rows.append((base_label, m.as_row()))
        payloads[base_label] = metrics_payload(m)
        print(f"[eval] base win={m.wins}/{m.n_episodes}", flush=True)

    # ---- 訓練後模型 ----
    tuned_eps: list[EpisodeStats] | None = None
    if args.adapter:
        print(f"[eval] 跑訓練後模型（adapter={args.adapter}）…", flush=True)
        if shared_vllm_backend is not None:
            shared_vllm_backend.adapter = args.adapter
            tuned_eps = run_llm_with_backend(args, shared_vllm_backend, words=words, legal=legal)
            del shared_vllm_backend
        else:
            tuned_eps = run_llm(args, adapter=args.adapter, words=words, legal=legal)
        m = aggregate(tuned_eps)
        rows.append((args.label_tuned, m.as_row()))
        payloads[args.label_tuned] = metrics_payload(m)
        print(
            f"[eval] tuned win={m.wins}/{m.n_episodes} "
            f"({100 * m.win_rate:.1f}% CI [{100 * m.win_ci_low:.1f}, {100 * m.win_ci_high:.1f}])",
            flush=True,
        )

    # ---- transcript 挑選 ----
    transcript_section: list[str] = []
    if tuned_eps:
        args.transcripts_dir.mkdir(parents=True, exist_ok=True)
        for i, e in enumerate(pick_transcripts(tuned_eps), 1):
            md = e.transcript_markdown()
            (args.transcripts_dir / f"{i:02d}_{e.answer}.md").write_text(md, encoding="utf-8")
            transcript_section.append(md)

    # ---- 報告 ----
    lines = [
        "# 最終評測報告（真實執行結果）",
        "",
        f"- 評測集：固定 {args.split}（seed={args.seed}，n={len(words)}），train/eval 嚴格隔離",
        f"- 解碼：greedy（do_sample=False），max_new_tokens={args.max_new_tokens}，"
        f"線索摘要={'關' if args.no_summary else '開'}",
        f"- 產生時間：{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "| agent | " + " | ".join(h for _, h in METRIC_COLUMNS) + " |",
        "|" + "---|" * (len(METRIC_COLUMNS) + 1),
    ]
    for label, row in rows:
        lines.append(
            "| " + label + " | " + " | ".join(row.get(k, "—") for k, _ in METRIC_COLUMNS) + " |"
        )
    lines += [
        "",
        "> 指標口徑同 baselines.md；heuristic 具答案表存取（參照上界），不可與 LLM 直接互比。",
        "",
    ]
    if transcript_section:
        lines += ["## 代表性對局（訓練後模型，含失敗案例）", ""] + transcript_section
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")

    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "meta": {
                    "n": len(words),
                    "seed": args.seed,
                    "split": args.split,
                },
                "rows": payloads,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"[eval] 報告已寫入 {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
