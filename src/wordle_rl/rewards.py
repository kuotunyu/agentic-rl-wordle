"""獎勵函數：EpisodeStats 的純函數，零 torch、零 trainer 依賴。

量級設計（詳細論證見 docs/rewards.md）：
- 獲勝 +10 + 1×(6−turns_used) ∈ [10,15]：快勝加成嚴格遞減 → 拖延無利可圖。
- shaping 只獎「首次發現」：新綠位 +0.2（≤5 個 → ≤1.0）、新存在單位 +0.1（≤5 → ≤0.5）。
  shaping 總上界 1.5 << 最小勝利獎勵 10 → 「刷 shaping 不求勝」被 ≥8.5 的差距支配。
- 非法/格式錯 −2/回合：大於單回合最大 shaping（+0.7 = 首猜全綠 0.2×2+0.1×...，實務上限）。
- 違反已知限制 −1/回合：溫和——探測性猜詞（故意用排除字母收集資訊）是合法策略，輕罰不禁止。
- 重複同字 −2/次：重複零資訊且必虧（防 reward hacking 的下限保證）。

BINARY preset（勝 1 / 敗 0）：HF 官方 Wordle 範例發現純輸贏優於 shaped——保留為 A/B 對照。
GRPO 每條軌跡一個 advantage → 各項相加成單一 episode 純量是正確形態。
"""

from __future__ import annotations

from dataclasses import dataclass

from .episode import EpisodeStats


@dataclass(frozen=True)
class RewardConfig:
    max_turns: int = 6
    win_base: float = 10.0
    win_turn_bonus: float = 1.0      # ×(max_turns − turns_used)
    new_green: float = 0.2
    new_presence: float = 0.1
    illegal_penalty: float = -2.0    # 每個非法/格式錯回合
    violation_penalty: float = -1.0  # 每個含違規的回合
    repeat_penalty: float = -2.0     # 每次重複同字

DEFAULT_CONFIG = RewardConfig()

# shaping 理論上界（供測試與文件斷言）：5 綠 ×0.2 + 5 存在 ×0.1
MAX_SHAPING = 5 * DEFAULT_CONFIG.new_green + 5 * DEFAULT_CONFIG.new_presence  # = 1.5


def shaped_reward(stats: EpisodeStats, cfg: RewardConfig = DEFAULT_CONFIG) -> float:
    r = 0.0
    if stats.won:
        r += cfg.win_base + cfg.win_turn_bonus * (cfg.max_turns - stats.turns_used)
    r += cfg.new_green * stats.total_new_greens
    r += cfg.new_presence * stats.total_new_presence
    r += cfg.illegal_penalty * stats.num_illegal
    r += cfg.violation_penalty * stats.num_violation_turns
    r += cfg.repeat_penalty * stats.num_repeats
    return round(r, 6)


def binary_reward(stats: EpisodeStats) -> float:
    return 1.0 if stats.won else 0.0


def episode_reward(stats: EpisodeStats, preset: str, cfg: RewardConfig = DEFAULT_CONFIG) -> float:
    if preset == "shaped":
        return shaped_reward(stats, cfg)
    if preset == "binary":
        return binary_reward(stats)
    raise ValueError(f"未知 reward preset：{preset!r}（可用：shaped / binary）")
