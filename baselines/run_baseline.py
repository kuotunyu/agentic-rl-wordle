"""階段 1：在固定 200 個 eval 詞上量測 baseline。

    python baselines/run_baseline.py --agent random
    python baselines/run_baseline.py --agent heuristic
    python baselines/run_baseline.py --agent llm --backend vllm      # Colab（GPU）
    python baselines/run_baseline.py --agent llm --backend transformers --n 5   # 本機小樣本煙霧

結果合併進 results/baselines.json（逐局原始記錄），並重新產生 results/baselines.md。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from wordle_rl.agents import HeuristicAgent, LLMAgent, RandomAgent
from wordle_rl.metrics import AggregateMetrics, aggregate
from wordle_rl.runner import run_episodes
from wordle_rl.words import get_splits, load_answers, load_legal

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "results"

METRIC_COLUMNS = [
    ("win_rate", "勝率 [95% CI]"),
    ("avg_guesses_on_wins", "勝局均猜"),
    ("illegal_rate", "非法輸出率"),
    ("tag_ok_rate", "tag 格式遵循率"),
    ("absent_reuse_rate", "重用已知不存在字母率"),
    ("green_break_rate", "破壞已知綠位率"),
    ("repeat_rate", "重複猜測率"),
]


def render_markdown(store: dict) -> str:
    meta = store["meta"]
    lines = [
        "# 階段 1：Baselines（真實執行結果）",
        "",
        f"- 評測集：固定 eval_200（seed={meta['seed']}，n={meta['n']}，train/eval 嚴格隔離）",
        f"- 環境：最多 {meta['max_turns']} 回合；非法猜測消耗回合",
        "- 指標定義：非法輸出率 =（非 tag_ok **或** 環境判非法）回合 / 全部回合（union 口徑）；",
        "  tag 格式遵循率單獨分列；違限率分母 = 已有線索的回合",
        "",
        "| agent | " + " | ".join(h for _, h in METRIC_COLUMNS) + " |",
        "|" + "---|" * (len(METRIC_COLUMNS) + 1),
    ]
    for name, entry in store["agents"].items():
        row = entry["metrics_row"]
        lines.append(f"| {name} | " + " | ".join(row.get(k, "—") for k, _ in METRIC_COLUMNS) + " |")
    lines += [
        "",
        "> ⚠ heuristic baseline 看得到完整答案表（含 eval 詞的分布）——這是 Wordle solver 的",
        "> 常規做法，作為參照上界使用；LLM agent 沒有這種存取，兩者不能直接互比。",
        "",
        f"_更新時間：{store['updated_at']}；逐局原始記錄見 `baselines.json`。_",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", required=True, choices=["random", "heuristic", "llm"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--backend", default="transformers", choices=["transformers", "vllm"])
    ap.add_argument("--adapter", default=None, help="LoRA adapter 路徑（評測訓練後模型用）")
    ap.add_argument("--label", default=None, help="結果表中的 agent 名稱（預設自動）")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42, help="切分 seed（勿改，紅線）")
    ap.add_argument("--agent-seed", type=int, default=0, help="random agent 的抽樣 seed")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--no-summary", action="store_true", help="關閉線索摘要（消融）")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    splits = get_splits(seed=args.seed)
    words = list(splits.eval_200[: args.n])
    legal = load_legal()

    if args.agent == "random":
        agent = RandomAgent(legal, seed=args.agent_seed)
        label = args.label or "random"
        config = {"agent_seed": args.agent_seed}
    elif args.agent == "heuristic":
        agent = HeuristicAgent(load_answers())
        label = args.label or "heuristic"
        config = {"candidate_pool": "answers(2315)"}
    else:
        from wordle_rl.backends import GenConfig

        if args.backend == "vllm":
            from wordle_rl.backends import VLLMBackend

            backend = VLLMBackend(args.model, adapter=args.adapter)
        else:
            from wordle_rl.backends import TransformersBackend

            backend = TransformersBackend(args.model, adapter=args.adapter)
        gen_cfg = GenConfig(max_new_tokens=args.max_new_tokens, do_sample=False, stop=("</guess>",))
        agent = LLMAgent(backend, gen_cfg, include_summary=not args.no_summary)
        label = args.label or (
            args.model.split("/")[-1].lower() + ("+lora" if args.adapter else "-base")
        )
        config = {
            "model": args.model,
            "backend": args.backend,
            "adapter": args.adapter,
            "greedy": True,
            "max_new_tokens": args.max_new_tokens,
            "include_summary": not args.no_summary,
        }

    print(f"[baseline] agent={label} n={len(words)} …", flush=True)
    episodes = run_episodes(agent, words, legal)
    metrics: AggregateMetrics = aggregate(episodes)
    print(
        f"[baseline] {label}: win={metrics.wins}/{metrics.n_episodes} "
        f"({100 * metrics.win_rate:.1f}% CI [{100 * metrics.win_ci_low:.1f}, {100 * metrics.win_ci_high:.1f}]) "
        f"avg_guesses(win)={metrics.avg_guesses_on_wins} illegal={100 * metrics.illegal_rate:.1f}%",
        flush=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "baselines.json"
    store = (
        json.loads(json_path.read_text(encoding="utf-8"))
        if json_path.exists()
        else {"meta": {}, "agents": {}}
    )
    store["meta"] = {"n": args.n, "seed": args.seed, "max_turns": 6}
    store["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    store["agents"][label] = {
        "config": config,
        "metrics_row": metrics.as_row(),
        "metrics": {
            "n": metrics.n_episodes,
            "wins": metrics.wins,
            "win_rate": metrics.win_rate,
            "win_ci": [metrics.win_ci_low, metrics.win_ci_high],
            "avg_guesses_on_wins": metrics.avg_guesses_on_wins,
            "illegal_rate": metrics.illegal_rate,
            "tag_ok_rate": metrics.tag_ok_rate,
            "absent_reuse_rate": metrics.absent_reuse_rate,
            "green_break_rate": metrics.green_break_rate,
            "repeat_rate": metrics.repeat_rate,
            "turns_with_info": metrics.turns_with_info,
            "total_turns": metrics.total_turns,
        },
        "episodes": [e.to_dict() for e in episodes],
    }
    json_path.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out / "baselines.md").write_text(render_markdown(store), encoding="utf-8")
    print(f"[baseline] 寫入 {json_path} 與 baselines.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
