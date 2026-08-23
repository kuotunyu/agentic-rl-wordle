"""parser 三態、線索摘要渲染、build_messages 的歷史不可變性。"""

import re

from wordle_rl.env.wordle import ILLEGAL, WORD_LEN, score_guess
from wordle_rl.episode import TurnRecord
from wordle_rl.knowledge import Knowledge
from wordle_rl.protocol import (
    _TURN_INSTRUCTION,
    SYSTEM_PROMPT,
    build_messages,
    canonical_assistant,
    parse_guess,
    render_constraint_summary,
)

# ---------- parse_guess ----------


def test_parse_tag_ok():
    r = parse_guess("I'll try a common word. <guess>crane</guess>")
    assert (r.guess, r.outcome) == ("crane", "tag_ok")


def test_parse_last_tag_wins():
    r = parse_guess("<guess>slate</guess> no wait <guess>crane</guess>")
    assert (r.guess, r.outcome) == ("crane", "tag_ok")


def test_parse_uppercase_and_whitespace():
    r = parse_guess("<guess>  CRANE </guess>")
    assert (r.guess, r.outcome) == ("crane", "tag_ok")


def test_parse_malformed_tag_word_is_not_rescued_as_the_literal_guess():
    """tag 字面 'guess' 是 5 字母詞——剝除標記後不得被 fallback 誤抓。"""
    r = parse_guess("<guess>slates</guess>")
    assert r.outcome == "no_parse" and r.guess is None


def test_parse_malformed_tag_salvaged_from_surrounding_text():
    r = parse_guess("<guess>slates</guess> ... I mean slate")
    assert (r.guess, r.outcome) == ("slate", "fallback_word")


def test_parse_fallback_last_five_letter_word():
    r = parse_guess("my first thought is crane but I'll go with slate")
    assert (r.guess, r.outcome) == ("slate", "fallback_word")


def test_parse_no_parse():
    r = parse_guess("I do not know.")
    assert (r.guess, r.outcome) == (None, "no_parse")


def test_format_example_placeholder_is_a_real_five_letter_word():
    """M3.2 pilot 實測發現：SYSTEM_PROMPT/_TURN_INSTRUCTION 曾用字面 'word'（4 字母）
    當格式範例，每回合重複出現，模型學到的是範例的「字母數」而非規則本身——
    transcript 裡大量 no_parse 猜測剛好是 4 字母詞（girl/zone/leak/pool/...），
    與 'word' 的長度可疑地一致。範例本身必須是真正的 5 字母詞，防止同類 regression。
    """
    for text in (SYSTEM_PROMPT, _TURN_INSTRUCTION):
        examples = re.findall(r"<guess>\s*([a-zA-Z]+)\s*</guess>", text)
        assert examples, f"找不到 <guess>...</guess> 範例：{text!r}"
        for word in examples:
            assert len(word) == WORD_LEN, f"格式範例 {word!r} 不是 {WORD_LEN} 字母，會誤導模型"


# ---------- 摘要與訊息渲染 ----------


def mk_record(turn, guess, feedback, raw=None, outcome="tag_ok", **kw):
    defaults = dict(
        turn=turn,
        raw_output=raw if raw is not None else f"<guess>{guess}</guess>",
        parse_outcome=outcome,
        guess=guess,
        feedback=feedback,
        illegal=feedback == ILLEGAL,
        illegal_reason=None,
        violations=(),
        had_info=turn > 1,
        is_repeat=False,
        new_greens=0,
        new_presence=0,
    )
    defaults.update(kw)
    return TurnRecord(**defaults)


def test_constraint_summary_content():
    k = Knowledge()
    k.update("arise", score_guess("arise", "crane"))  # YGXXG
    s = render_constraint_summary(k)
    assert "position 2 = r" in s
    assert "position 5 = e" in s
    assert "a (at least 1)" in s
    assert "not in word: i, s" in s
    assert "a is not at position 1" in s
    # 已由綠位完整呈現的字母不重複列在 contains
    assert "r (at least" not in s and "e (at least" not in s


def test_constraint_summary_empty_knowledge():
    assert render_constraint_summary(Knowledge()) == ""


def test_constraint_summary_exact_count_survives_green_coverage():
    """對抗性審查確認的缺陷：exactly N 是上界，綠位傳達不了，必須永遠明示。

    eerie vs crane → XXYXG：exact e=1 且 e 有 1 個綠位——舊邏輯會把
    「恰好一個 e」吃掉，讓 there/theme 這類雙 e 詞看似與摘要一致。
    """
    k = Knowledge()
    k.update("eerie", score_guess("eerie", "crane"))  # XXYXG
    s = render_constraint_summary(k)
    assert "e (exactly 1)" in s
    assert not k.is_consistent("there")  # 摘要現在與 tracker 一致地排除雙 e 詞

    k2 = Knowledge()
    k2.update("eerie", score_guess("eerie", "elite"))  # GXXYG：exact e=2、兩個綠 e
    s2 = render_constraint_summary(k2)
    assert "e (exactly 2)" in s2


def test_canonical_assistant():
    ok = mk_record(1, "crane", "GYXXY", raw="thinking... <guess>crane</guess>")
    assert canonical_assistant(ok) == "<guess>crane</guess>"
    bad = mk_record(1, None, ILLEGAL, raw="I have no idea", outcome="no_parse")
    assert canonical_assistant(bad) == "I have no idea"


def test_build_messages_empty_history():
    msgs = build_messages([])
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert "Turn 1 of 6" in msgs[1]["content"]


def test_build_messages_alternation_and_content():
    h = [
        mk_record(1, "arise", score_guess("arise", "crane")),
        mk_record(2, "route", score_guess("route", "crane")),
    ]
    msgs = build_messages(h)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[2]["content"] == "<guess>arise</guess>"
    assert 'You guessed "arise"' in msgs[3]["content"]
    assert "Turn 2 of 6" in msgs[3]["content"]
    assert "Turn 3 of 6" in msgs[5]["content"]


def test_build_messages_history_prefix_is_immutable():
    """第 k 回合的訊息只依賴前 k 回合——後續回合不得改寫先前訊息。"""
    h = [
        mk_record(1, "arise", score_guess("arise", "crane")),
        mk_record(2, "route", score_guess("route", "crane")),
    ]
    one = build_messages(h[:1])
    two = build_messages(h)
    assert one == two[: len(one)]


def test_build_messages_illegal_turn_rendering():
    h = [mk_record(1, "zzzzz", ILLEGAL, illegal_reason="not_in_word_list")]
    msgs = build_messages(h)
    assert "not a valid 5-letter English word" in msgs[3]["content"]
    assert "wasted" in msgs[3]["content"]
    h2 = [mk_record(1, None, ILLEGAL, raw="???", outcome="no_parse", illegal_reason="none")]
    msgs2 = build_messages(h2)
    assert "did not contain a valid" in msgs2[3]["content"]


def test_build_messages_summary_toggle():
    h = [mk_record(1, "arise", score_guess("arise", "crane"))]
    with_summary = build_messages(h, include_summary=True)
    without = build_messages(h, include_summary=False)
    assert "Known so far" in with_summary[3]["content"]
    assert "Known so far" not in without[3]["content"]
