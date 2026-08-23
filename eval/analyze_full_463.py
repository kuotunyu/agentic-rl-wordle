"""Validate and summarize the complete 463-word base-vs-GRPO evaluation.

The base model lost every episode, so each tuned-model win is necessarily a
base-loss/tuned-win discordant pair.  This makes the exact paired McNemar test
recoverable from the aggregate report without storing model generations.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

from wordle_rl.metrics import wilson_ci

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "full_463_report.json"
MAX_TURNS = 6


def exact_mcnemar_base_zero(tuned_wins: int) -> float:
    """Two-sided exact McNemar p-value when base has zero paired wins."""
    if tuned_wins < 0:
        raise ValueError("tuned_wins must be non-negative")
    if tuned_wins == 0:
        return 1.0
    return min(1.0, 2.0 * (0.5**tuned_wins))


def _rate_fraction(name: str, rate: float, max_denominator: int) -> Fraction:
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError(f"{name} must be a finite rate between zero and one")
    fraction = Fraction(rate).limit_denominator(max_denominator)
    if not math.isclose(float(fraction), rate, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{name} cannot be recovered as integer action counts "
            f"with denominator <= {max_denominator}"
        )
    return fraction


def _unique_denominator(
    fractions: tuple[Fraction, ...], minimum: int, maximum: int, label: str
) -> int:
    base = math.lcm(*(fraction.denominator for fraction in fractions))
    first_multiplier = max(1, math.ceil(minimum / base))
    last_multiplier = maximum // base
    candidates = [base * multiplier for multiplier in range(first_multiplier, last_multiplier + 1)]
    if len(candidates) != 1:
        raise ValueError(f"rates do not identify a unique integer action denominator for {label}")
    return candidates[0]


def _recover_action_counts(row: dict, n_episodes: int) -> dict[str, int]:
    max_turns = n_episodes * MAX_TURNS
    turn_rates = {
        "illegal": _rate_fraction("illegal_rate", row["illegal_rate"], max_turns),
        "protocol_adherent": _rate_fraction("tag_ok_rate", row["tag_ok_rate"], max_turns),
        "repeats": _rate_fraction("repeat_rate", row["repeat_rate"], max_turns),
    }
    total_turns = _unique_denominator(
        tuple(turn_rates.values()), n_episodes, max_turns, "total turns"
    )

    counts = {name: int(fraction * total_turns) for name, fraction in turn_rates.items()}
    counts["legal"] = total_turns - counts.pop("illegal")

    info_rates = {
        "absent_reuses": _rate_fraction("absent_reuse_rate", row["absent_reuse_rate"], total_turns),
        "green_breaks": _rate_fraction("green_break_rate", row["green_break_rate"], total_turns),
    }
    turns_with_info = _unique_denominator(
        tuple(info_rates.values()), 1, total_turns, "information turns"
    )
    counts.update({name: int(fraction * turns_with_info) for name, fraction in info_rates.items()})

    return {
        "total_turns": total_turns,
        "protocol_adherent": counts["protocol_adherent"],
        "legal": counts["legal"],
        "repeats": counts["repeats"],
        "turns_with_info": turns_with_info,
        "absent_reuses": counts["absent_reuses"],
        "green_breaks": counts["green_breaks"],
    }


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
        raise ValueError("aggregate counts cannot recover a paired McNemar test when base has wins")

    for label, row in (("base", base), ("tuned", tuned)):
        expected_rate = row["wins"] / row["n"]
        if not math.isclose(row["win_rate"], expected_rate, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"{label} win_rate does not match wins/n")

    exact_p = exact_mcnemar_base_zero(tuned["wins"])
    corrected_p = min(1.0, exact_p * sequential_looks)
    tuned_action_counts = _recover_action_counts(tuned, tuned["n"])
    base_ci = wilson_ci(base["wins"], base["n"])
    tuned_ci = wilson_ci(tuned["wins"], tuned["n"])
    return {
        "schema_version": 2,
        "source_evidence": "results/full_463_report.json (committed aggregate)",
        "source_evidence_level": "aggregate; full per-episode transcripts unavailable",
        "split": "eval_full",
        "seed": meta["seed"],
        "n_pairs": 463,
        "base": base_name,
        "tuned": tuned_name,
        "base_wins": base["wins"],
        "tuned_wins": tuned["wins"],
        "absolute_win_rate_gain": tuned["win_rate"] - base["win_rate"],
        "base_win_rate_wilson_ci_95": base_ci,
        "tuned_win_rate_wilson_ci_95": tuned_ci,
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
            None if tuned["absent_reuse_rate"] is None else 1.0 - tuned["absent_reuse_rate"]
        ),
        "tuned_green_preservation_rate": (
            None if tuned["green_break_rate"] is None else 1.0 - tuned["green_break_rate"]
        ),
        "tuned_repeat_rate": tuned["repeat_rate"],
        "tuned_action_counts": tuned_action_counts,
        "interpretation": ("statistically_significant_but_small_practical_task_success"),
    }


def render_markdown(result: dict) -> str:
    base_low, base_high = result["base_win_rate_wilson_ci_95"]
    tuned_low, tuned_high = result["tuned_win_rate_wilson_ci_95"]
    counts = result["tuned_action_counts"]
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
            (f"- Observed absolute win-rate gain: **{result['absolute_win_rate_gain']:.1%}**"),
            (
                f"- Wilson 95% win-rate intervals: base "
                f"[{base_low:.1%}, {base_high:.1%}]; GRPO "
                f"[{tuned_low:.1%}, {tuned_high:.1%}]"
            ),
            (f"- Exact paired McNemar: `p={result['mcnemar_two_sided_exact_p']:.6f}`"),
            (
                f"- Conservative correction for {result['sequential_looks']} "
                f"nested looks (n=200, then n=463): Bonferroni "
                f"`p={result['bonferroni_corrected_p']:.6f}`"
            ),
            (
                f"- Recovered turn counts: protocol-adherent "
                f"{counts['protocol_adherent']}/{counts['total_turns']}; legal "
                f"{counts['legal']}/{counts['total_turns']}; repeats "
                f"{counts['repeats']}/{counts['total_turns']}"
            ),
            (
                f"- Recovered information-turn counts: absent-letter reuse "
                f"{counts['absent_reuses']}/{counts['turns_with_info']}; green-position "
                f"breaks {counts['green_breaks']}/{counts['turns_with_info']}"
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
            (f"| Win within 6 turns | 0.0% | **{result['absolute_win_rate_gain']:.1%}** |"),
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
            (
                "Evidence boundary: these values are recomputed from the committed "
                "aggregate JSON. Full per-episode records are not committed, so the "
                "per-turn source rows cannot be independently re-aggregated."
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
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed analysis files match recomputation without rewriting",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sequential_looks <= 0:
        raise ValueError("--sequential-looks must be positive")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(payload, sequential_looks=args.sequential_looks)
    md_path = args.output_prefix.with_suffix(".md")
    json_path = args.output_prefix.with_suffix(".json")
    markdown = render_markdown(result)
    json_text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        mismatches = []
        if not md_path.exists() or md_path.read_text(encoding="utf-8") != markdown:
            mismatches.append(str(md_path))
        if not json_path.exists() or json_path.read_text(encoding="utf-8") != json_text:
            mismatches.append(str(json_path))
        if mismatches:
            print("Analysis artifacts are stale: " + ", ".join(mismatches))
            return 1
        print("Analysis artifacts match committed aggregate recomputation.")
        return 0

    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json_text, encoding="utf-8")
    print(markdown)
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
