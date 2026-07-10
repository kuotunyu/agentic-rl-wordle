"""Knowledge 線索追蹤：計數語義（重複字母陷阱）、違規判定、一致性過濾、shaping delta。"""

from wordle_rl.env.wordle import ILLEGAL, score_guess
from wordle_rl.knowledge import Knowledge


def fed(k: Knowledge, guess: str, answer: str):
    return k.update(guess, score_guess(guess, answer))


def test_plain_x_means_absent():
    k = Knowledge()
    fed(k, "speed", "dolly")  # s,p,e 全不在 dolly；d 在 → XXXXY
    assert k.exact_counts["s"] == 0
    assert k.exact_counts["p"] == 0
    assert k.exact_counts["e"] == 0
    assert k.min_counts["d"] == 1
    assert "d" not in k.exact_counts


def test_duplicate_x_is_exact_count_not_absent():
    """經典陷阱：dolly 猜 lolly → 開頭 l 拿 X，但 l 明明存在（恰 2 個）。"""
    k = Knowledge()
    fed(k, "lolly", "dolly")  # XGGGG
    assert k.exact_counts["l"] == 2      # X 出現 ⇒ 數量探底 = g+y = 2
    assert k.min_counts["l"] == 2
    # 下一手用 l 絕不可判「重用不存在字母」
    assert "reuses_absent_letter" not in k.violations("llama")


def test_violation_reuses_absent_letter():
    k = Knowledge()
    fed(k, "speed", "dolly")  # s/p/e exact 0
    assert k.violations("spare") == {"reuses_absent_letter"}   # s、p、e 都已知不存在
    assert k.violations("dolly") == set()


def test_violation_breaks_green():
    k = Knowledge()
    fed(k, "arise", "crane")  # YGXXG：greens {1:r, 4:e}
    assert k.greens == {1: "r", 4: "e"}
    v = k.violations("route")  # pos1 是 o ≠ r
    assert "breaks_green" in v
    assert k.violations("crane") == set()


def test_illegal_feedback_is_noop():
    k = Knowledge()
    delta = k.update("zzzzz", ILLEGAL)
    assert delta.new_greens == 0 and delta.new_presence == 0
    assert not k.has_info()


def test_update_delta_first_discoveries_and_upgrade():
    k = Knowledge()
    d1 = fed(k, "arise", "crane")  # YGXXG：r、e 綠，a 黃
    assert d1.new_greens == 2
    assert d1.new_presence == 3    # r、e、a 各 +1 存在單位
    d2 = fed(k, "crane", "crane")  # 全綠
    assert d2.new_greens == 3      # c、a、n 三個新綠位（a 是 Y→G 升級）
    assert d2.new_presence == 2    # c、n 是新發現；a/r/e 的存在已知
    # delta 上界：綠 ≤5、存在 ≤5
    assert len(k.greens) == 5 and sum(k.min_counts.values()) == 5


def test_repeated_guess_gives_zero_delta():
    k = Knowledge()
    fed(k, "arise", "crane")
    d = fed(k, "arise", "crane")
    assert d.new_greens == 0 and d.new_presence == 0


def test_exact_count_from_partial_duplicate():
    k = Knowledge()
    fed(k, "speed", "abide")  # XXYXY：e 得 Y+X ⇒ exact e = 1
    assert k.exact_counts["e"] == 1
    assert k.min_counts["e"] == 1
    assert not k.is_consistent("eerie")   # 3 個 e ≠ 1
    assert k.is_consistent("abide")


def test_is_consistent_full_semantics():
    k = Knowledge()
    fed(k, "label", "hello")  # YXXYY：l min 2、e min 1、a/b exact 0、not_at 各位
    assert k.min_counts["l"] == 2
    assert k.exact_counts["a"] == 0 and k.exact_counts["b"] == 0
    assert k.is_consistent("hello")
    assert not k.is_consistent("llama")   # 缺 e、含 a
    assert not k.is_consistent("lolly")   # 缺 e
    # not_at：label 的 pos0 l 拿 Y ⇒ l 不在 0 → "l...." 開頭皆不一致
    assert not k.is_consistent("leale")


def test_not_at_from_x_and_y():
    k = Knowledge()
    fed(k, "arise", "crane")  # YGXXG
    assert "a" in k.not_at[0]   # Y 也證明不在該位
    assert "i" in k.not_at[2]   # X 同樣證明不在該位
    assert 1 not in k.not_at or "r" not in k.not_at.get(1, set())


def test_has_info():
    k = Knowledge()
    assert not k.has_info()
    fed(k, "speed", "dolly")
    assert k.has_info()
