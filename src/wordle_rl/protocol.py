"""Agent 協定：system prompt、對話渲染、robust parser。

- 協定語言為英文（英文單字遊戲對 1.5B 模型較友善）；文件中文。
- 過去 assistant turn 正規化為 `<guess>word</guess>`（原始輸出只進 log）：
  控制 context 膨脹、去除過期思考。解析失敗的 turn 保留原始輸出（真實呈現）。
- baselines / eval / 訓練 rollout 全部共用 build_messages —— 公平比較的前提。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .env.wordle import ILLEGAL, WORD_LEN

if TYPE_CHECKING:  # 避免執行期循環 import
    from .episode import TurnRecord
    from .knowledge import Knowledge

SYSTEM_PROMPT = """You are playing Wordle. Find the secret 5-letter English word within 6 guesses.

After each guess you receive feedback, one mark per letter position:
- G: this letter is in the word at exactly this position.
- Y: this letter is in the word but at a different position.
- X: this letter is not in the word. If your guess repeats a letter, G/Y marks are limited by how many times that letter appears in the secret word, and the extra copies get X.

Rules:
- Every guess must be a valid 5-letter English word.
- An invalid word or a badly formatted reply wastes a turn.
- Think briefly if needed (at most two short sentences), then output exactly one guess formatted as <guess>word</guess>. Never output more than one <guess> tag."""

_TURN_INSTRUCTION = "Turn {turn} of {max_turns}. Output your guess as <guess>word</guess>."

_GUESS_TAG_RE = re.compile(r"<guess>\s*([a-zA-Z]+)\s*</guess>", re.IGNORECASE)
_TAG_MARKUP_RE = re.compile(r"</?\s*guess\s*>", re.IGNORECASE)
_WORD5_RE = re.compile(r"\b[a-zA-Z]{5}\b")


@dataclass(frozen=True)
class ParseResult:
    guess: str | None
    outcome: str  # "tag_ok" | "fallback_word" | "no_parse"


def parse_guess(text: str) -> ParseResult:
    """三態解析：最後一個 <guess> tag → 全文最後一個獨立 5 字母詞 → 放棄。

    fallback 掃描前先剝除 <guess>/</guess> 標記——tag 字面本身含 5 字母單字
    "guess"，不剝除的話 malformed tag 會被誤抓成猜測。
    """
    tags = _GUESS_TAG_RE.findall(text)
    if tags and len(tags[-1]) == WORD_LEN:
        return ParseResult(guess=tags[-1].lower(), outcome="tag_ok")
    cleaned = _TAG_MARKUP_RE.sub(" ", text)
    words = _WORD5_RE.findall(cleaned)
    if words:
        return ParseResult(guess=words[-1].lower(), outcome="fallback_word")
    return ParseResult(guess=None, outcome="no_parse")


def render_constraint_summary(k: "Knowledge") -> str:
    """把已知限制渲染成一行英文（給模型看的位置用 1-based）。空知識回空字串。"""
    parts: list[str] = []

    if k.greens:
        greens = "; ".join(
            f"position {i + 1} = {letter}" for i, letter in sorted(k.greens.items())
        )
        parts.append(greens)

    contains: list[str] = []
    green_counts: dict[str, int] = {}
    for letter in k.greens.values():
        green_counts[letter] = green_counts.get(letter, 0) + 1
    for letter in sorted(set(k.min_counts) | {c for c, n in k.exact_counts.items() if n > 0}):
        exact = k.exact_counts.get(letter)
        if exact is not None and exact > 0:
            # 上界資訊綠位傳達不了（綠位只是下界）——永遠明示 exactly N，
            # 否則重複字母殘局的關鍵限制會從摘要消失（對抗性審查確認的缺陷）
            contains.append(f"{letter} (exactly {exact})")
            continue
        known = k.min_counts.get(letter, 0)
        if known <= green_counts.get(letter, 0):
            continue  # 純下界且已由綠位完整呈現 → 不重複列
        contains.append(f"{letter} (at least {known})")
    if contains:
        parts.append("contains: " + ", ".join(contains))

    absent = sorted(c for c, n in k.exact_counts.items() if n == 0)
    if absent:
        parts.append("not in word: " + ", ".join(absent))

    # 位置排除：只列「已知存在」的字母（對可解性最有用），且略過已綠化的位置
    present = {c for c, n in k.min_counts.items() if n > 0}
    pos_notes: list[str] = []
    for letter in sorted(present):
        positions = sorted(
            i + 1 for i, excluded in k.not_at.items() if letter in excluded and i not in k.greens
        )
        if positions:
            pos_str = " or ".join(str(p) for p in positions)
            pos_notes.append(f"{letter} is not at position {pos_str}")
    if pos_notes:
        parts.append("; ".join(pos_notes))

    return ("Known so far: " + "; ".join(parts) + ".") if parts else ""


def render_feedback_msg(
    record: "TurnRecord",
    knowledge: "Knowledge",
    turn_next: int,
    max_turns: int,
    include_summary: bool = True,
) -> str:
    """把上一回合的結果渲染成下一個 user turn 的文字。

    knowledge 必須是「已吸收 record 本身回饋」當下的知識快照——
    不可傳入更晚回合的知識，否則會改寫模型早先實際看到的訊息。
    """
    lines: list[str] = []
    if record.feedback == ILLEGAL:
        if record.guess is None:
            lines.append(
                "Your reply did not contain a valid <guess>word</guess> tag. This turn is wasted."
            )
        else:
            lines.append(
                f'"{record.guess}" is not a valid 5-letter English word. This turn is wasted.'
            )
    else:
        spaced = " ".join(record.feedback)
        lines.append(f'You guessed "{record.guess}" -> feedback: {spaced}.')
        if include_summary:
            summary = render_constraint_summary(knowledge)
            if summary:
                lines.append(summary)
    lines.append(_TURN_INSTRUCTION.format(turn=turn_next, max_turns=max_turns))
    return "\n".join(lines)


def canonical_assistant(record: "TurnRecord") -> str:
    """歷史 assistant turn 的正規化文字：可解析 → 純 tag；不可解析 → 原始輸出。"""
    if record.guess is not None:
        return f"<guess>{record.guess}</guess>"
    return record.raw_output


def build_messages(
    history: list["TurnRecord"],
    max_turns: int = 6,
    include_summary: bool = True,
) -> list[dict]:
    """組出當前回合要送給模型的完整對話（system + user + 交替歷史）。

    知識摘要由內部沿 history 逐回合重建——第 k 回合的 user 訊息只依賴
    前 k 回合的記錄，因此同一 history 前綴的渲染跨回合恆定（歷史不可變、
    vLLM prefix cache 可命中）。呼叫端不需要（也不可以）傳入即時知識。
    """
    from .knowledge import Knowledge  # 延遲 import，避免模組層循環

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Guess the secret 5-letter word. "
            + _TURN_INSTRUCTION.format(turn=1, max_turns=max_turns),
        },
    ]
    k = Knowledge()
    for idx, record in enumerate(history):
        messages.append({"role": "assistant", "content": canonical_assistant(record)})
        if record.guess is not None and record.feedback != ILLEGAL:
            k.update(record.guess, record.feedback)
        messages.append(
            {
                "role": "user",
                "content": render_feedback_msg(
                    record,
                    k,
                    turn_next=idx + 2,
                    max_turns=max_turns,
                    include_summary=include_summary,
                ),
            }
        )
    return messages
