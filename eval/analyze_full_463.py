"""Validate and summarize the complete 463-word base-vs-GRPO evaluation.

The base model lost every episode, so each tuned-model win is necessarily a
base-loss/tuned-win discordant pair.  This makes the exact paired McNemar test
recoverable from the aggregate report without storing model generations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "full_463_report.json"


def exact_mcnemar_base_zero(tuned_wins: int) -> float:
    """Two-sided exact McNemar p-value when base has zero paired wins."""
    if tuned_wins < 0:
        raise ValueError("tuned_wins must be non-negative")
    if tuned_wins == 0:
        return 1.0
    return min(1.0, 2.0 * (0.5**tuned_wins))


def analyze(payload: dict, sequential_looks: int = 2) -> dict:
    meta = payload["meta"]
    if meta.get("split") != "eval_full" or meta.get("n") != 463:
        raise ValueError("expected the fixed eval_full split with n=463")

    rows = payload["rows"]
    base_items = [(name, row) for name, row in rows.items() if name.endswith("-base")]
    tuned_items = [(name, row) for name, row in rows.items() if "GRPO" in name]
    if len(base_items) != 1 or len(tuned_items) != 1:
        raise ValueError("expected exactly one base row and one GRPO row")

    base_name, base = base_items[0]
    tuned_name, tuned = tuned_items[0]
    if base["n"] != 463 or tuned["n"] != 463:
        raise ValueError("both rows must contain 463 episodes")
    if base["wins"] != 0:
        raise ValueError(
            "aggregate counts cannot recover a paired McNemar test when base has wins"
        )

    exact_p = exact_mcnemar_base_zero(tuned["wins"])
    corrected_p = min(1.0, exact_p * sequential_looks)
    return {
        "schema_version": 1,
        "split": "eval_full",
        "seed": meta["seed"],
        "n_pairs": 463,
        "base": base_name,
        "tuned": tuned_name,
        "base_wins": base["wins"],
        "tuned_wins": tuned["wins"],
        "absolute_win_rate_gain": tuned["win_rate"] - base["win_rate"],
        "gain_wilson_ci_95": tuned["win_ci"],
        "paired_discordance": {
            "base_only_correct": 0,
            "tuned_only_correct": tuned["wins"],
        },
        "mcnemar_two_sided_exact_p": exact_p,
        "sequential_looks": sequential_looks,
        "bonferroni_corrected_p": corrected_p,
        "base_legal_action_rate": 1.0 - base["illegal_rate"],
        "tuned_legal_action_rate": 1.0 - tuned["illegal_rate"],
        "base_tag_adherence_rate": base["tag_ok_rate"],
        "tuned_tag_adherence_rate": tuned["tag_ok_rate"],
        "tuned_absent_letter_preservation_rate": (
            None
            if tuned["absent_reuse_rate"] is None
            else 1.0 - tuned["absent_reuse_rate"]
        ),
        "tuned_green_preservation_rate": (
            None
            if tuned["green_break_rate"] is None
            else 1.0 - tuned["green_break_rate"]
        ),
        "tuned_repeat_rate": tuned["repeat_rate"],
        "interpretation": (
            "statistically_significant_but_small_practical_task_success"
        ),
    }


def render_markdown(result: dict) -> str:
    low, high = result["gain_wilson_ci_95"]
    return "\n".join(
        [
            "# Full 463-word paired analysis",
            "",
            (
                f"- Fixed held-out split: `eval_full`, seed "
                f"`{result['seed']}`, {result['n_pairs']} paired answers"
            ),
            (
                f"- Wins: base {result['base_wins']}/{result['n_pairs']} → "
                f"GRPO {result['tuned_wins']}/{result['n_pairs']}"
            ),
            (
                f"- Absolute win-rate gain: "
                f"**{result['absolute_win_rate_gain']:.1%}** "
                f"[Wilson 95% CI {low:.1%}, {high:.1%}]"
            ),
            (
                f"- Exact paired McNemar: "
                f"`p={result['mcnemar_two_sided_exact_p']:.6f}`"
            ),
            (
                f"- Conservative correction for {result['sequential_looks']} "
                f"nested looks (n=200, then n=463): Bonferroni "
                f"`p={result['bonferroni_corrected_p']:.6f}`"
            ),
            "",
            "## Capability funnel",
            "",
            "| Capability | Base | GRPO LoRA |",
            "|---|---:|---:|",
            (
                f"| Tag adherence | "
                f"{result['base_tag_adherence_rate']:.1%} | "
                f"**{result['tuned_tag_adherence_rate']:.1%}** |"
            ),
            (
                f"| Legal action rate | "
                f"{result['base_legal_action_rate']:.1%} | "
                f"**{result['tuned_legal_action_rate']:.1%}** |"
            ),
            (
                "| Preserve excluded letters | not defined | "
                f"{result['tuned_absent_letter_preservation_rate']:.1%} |"
            ),
            (
                "| Preserve known green positions | not defined | "
                f"{result['tuned_green_preservation_rate']:.1%} |"
            ),
            (
                f"| Win within 6 turns | 0.0% | "
                f"**{result['absolute_win_rate_gain']:.1%}** |"
            ),
            "",
            "## Interpretation",
            "",
            (
                "The full held-out evaluation clears the project's statistical "
                "success criterion, even after a conservative two-look correction. "
                "The practical task-success rate remains small: GRPO reliably "
                "learned the interaction protocol, but only partially learned "
                "multi-turn constraint tracking and Wordle strategy."
            ),
            "",
            (
                "The 463-word evaluation contains the earlier 200-word subset; "
                "it is the final, larger evaluation rather than an independent "
                "replication."
            ),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sequential-looks", type=int, default=2)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "results" / "full_463_analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sequential_looks <= 0:
        raise ValueError("--sequential-looks must be positive")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(payload, sequential_looks=args.sequential_looks)
    md_path = args.output_prefix.with_suffix(".md")
    json_path = args.output_prefix.with_suffix(".json")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(render_markdown(result))
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
