"""本機即時觀看 agent 解一個指定的詞：逐回合印出模型原始輸出、解析結果與彩色回饋。

    python play.py --answer crane                        # 未訓練 base 模型
    python play.py --answer crane --adapter runs/full/final
    python play.py                                       # 從 eval_200 隨機抽一個詞

裝置自動偵測（CUDA 優先，CPU 亦可跑但一回合約需數十秒）。
"""

from __future__ import annotations

import argparse
import random
import sys

from wordle_rl.backends import GenConfig, TransformersBackend
from wordle_rl.env.wordle import FEEDBACK_WIN, ILLEGAL, WordleEnv
from wordle_rl.knowledge import Knowledge
from wordle_rl.protocol import build_messages, render_constraint_summary
from wordle_rl.runner import EpisodeState, play_turn
from wordle_rl.words import get_splits, load_legal

try:
    from colorama import Back, Fore, Style, init as colorama_init

    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False


def tiles(guess: str | None, feedback: str, use_color: bool) -> str:
    if feedback == ILLEGAL or guess is None:
        return "✗ ILLEGAL（浪費一回合）"
    if not use_color:
        return f"{guess.upper()}  {' '.join(feedback)}"
    out = []
    for ch, mark in zip(guess.upper(), feedback):
        style = {
            "G": Back.GREEN + Fore.BLACK,
            "Y": Back.YELLOW + Fore.BLACK,
            "X": Back.WHITE + Fore.BLACK,
        }[mark]
        out.append(f"{style} {ch} {Style.RESET_ALL}")
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answer", default=None, help="5 字母答案；省略則從 eval_200 抽")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--device", default=None, help="auto/cpu/cuda（預設自動）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    use_color = HAS_COLOR and not args.no_color
    if use_color:
        colorama_init()

    legal = load_legal()
    answer = args.answer or random.Random(args.seed).choice(get_splits().eval_200)
    answer = answer.strip().lower()

    print(f"載入模型 {args.model}" + (f" + adapter {args.adapter}" if args.adapter else "") + " …")
    backend = TransformersBackend(
        args.model, adapter=args.adapter, device=None if args.device in (None, "auto") else args.device
    )
    cfg = GenConfig(max_new_tokens=args.max_new_tokens, do_sample=False, stop=("</guess>",))

    env = WordleEnv(legal=legal)
    env.reset(answer)
    state = EpisodeState(answer=answer, env=env, knowledge=Knowledge())

    print(f"\n=== Wordle：答案 [{answer.upper()}]（agent 看不到）===\n")
    while not state.done:
        chats = [build_messages(state.history, include_summary=not args.no_summary)]
        raw = backend.generate(chats, cfg)[0]
        record = play_turn(state, raw)
        print(f"─── 回合 {record.turn}/6 ───")
        print(f"模型輸出：{raw.strip()!r}")
        print(f"解析：{record.guess!r}（{record.parse_outcome}）"
              + (f"  ⚠ {'、'.join(record.violations)}" if record.violations else "")
              + ("  ⚠ 重複猜測" if record.is_repeat else ""))
        print(f"回饋：{tiles(record.guess, record.feedback, use_color)}")
        summary = render_constraint_summary(state.knowledge)
        if summary:
            print(f"線索：{summary}")
        print()

    if state.history[-1].feedback == FEEDBACK_WIN:
        print(f"🎉 {len(state.history)} 回合獲勝！")
    else:
        print(f"❌ 6 回合未中，答案是 {answer.upper()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
