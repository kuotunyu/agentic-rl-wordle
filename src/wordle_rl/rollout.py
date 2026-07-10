"""★ 唯一接觸 TRL 的模組：多輪 rollout 引擎 + rollout_func / reward_fn 轉接。

分層設計（可換 ART = 只換本模組）：
- play_batch_episodes(generate, games, …)：純 Python lockstep 引擎——所有 active episode
  每回合湊一批呼叫一次 generate；每局串接各回合的 token/logprobs。**本機可測**（fake generate）。
- make_rollout_func(...)：TRL GRPOTrainer 的 rollout_func 轉接。
- make_trl_reward_fn(...)：讀 rollout_func 轉發的 episode 統計欄位 → rewards.py 純函數。

關鍵語義（與 TRL 1.8 官方 openenv Wordle 範例同構）：
- assistant-only loss 是「結構性」的：只有模型生成的 token 進 completion_ids，
  環境回饋只出現在重新渲染的 prompt 側，永不進 loss。
- logprobs 是 vLLM 取樣當下的值；跨回合串接 = importance-sampling 近似（官方範例同）。
- 答案取樣在 rollout_func 內部（seeded RNG）：同一 prompt 槽位的 num_generations 局
  共用同一個隱藏答案 → GRPO 組內比較成立。dataset 的 prompt 內容不需要攜帶答案。
"""

from __future__ import annotations

import random
import statistics
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from .episode import EpisodeStats
from .rewards import RewardConfig, episode_reward
from .env.number_guess import number_guess_reward
from .games import TurnBasedGame


# ---------------------------------------------------------------- 核心引擎

@dataclass(frozen=True)
class TurnGen:
    """一個 episode 一回合的生成結果。len(token_ids) 必須 == len(logprobs)。"""

    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    text: str


GenerateFn = Callable[[list[list[dict]]], list[TurnGen]]


@dataclass
class EpisodeRollout:
    completion_ids: list[int]
    logprobs: list[float]
    stats: dict
    transcript: str
    budget_truncated: bool = False


def play_batch_episodes(
    generate: GenerateFn,
    games: list[TurnBasedGame],
    per_turn_max_tokens: int,
    max_total_tokens: int | None = None,
    max_turns_hard_cap: int = 16,
) -> list[EpisodeRollout]:
    """回合 lockstep：每輪對所有未結束（且預算足夠）的局發一次批次生成。"""
    n = len(games)
    completions: list[list[int]] = [[] for _ in range(n)]
    logps: list[list[float]] = [[] for _ in range(n)]
    truncated = [False] * n

    for _ in range(max_turns_hard_cap):
        active: list[int] = []
        for i, g in enumerate(games):
            if g.done:
                continue
            if max_total_tokens is not None and (
                len(completions[i]) + per_turn_max_tokens > max_total_tokens
            ):
                truncated[i] = True  # 預算不足以再生成一回合 → 視為敗局收場
                continue
            active.append(i)
        if not active:
            break

        chats = [games[i].build_messages() for i in active]
        gens = generate(chats)
        if len(gens) != len(active):
            raise RuntimeError(f"generate 回傳 {len(gens)} 筆，預期 {len(active)}")
        for i, gen in zip(active, gens):
            if len(gen.token_ids) != len(gen.logprobs):
                raise RuntimeError(
                    f"token/logprob 長度不齊：{len(gen.token_ids)} vs {len(gen.logprobs)}"
                )
            completions[i].extend(gen.token_ids)
            logps[i].extend(gen.logprobs)
            games[i].step(gen.text)

    out = []
    for i, g in enumerate(games):
        stats = g.stats()
        stats["budget_truncated"] = float(truncated[i])
        out.append(
            EpisodeRollout(
                completion_ids=completions[i],
                logprobs=logps[i],
                stats=stats,
                transcript=g.transcript(),
                budget_truncated=truncated[i],
            )
        )
    return out


# ---------------------------------------------------------------- 監控緩衝

class RolloutBuffers:
    """rollout_func / reward_fn 與 TrainerCallback 之間的共享緩衝（同進程）。"""

    def __init__(self, transcript_capacity: int = 64):
        self._lock = threading.Lock()
        self.transcripts: deque[str] = deque(maxlen=transcript_capacity)
        self.last_batch_stats: dict = {}
        self.last_reward_stats: dict = {}

    def record_rollouts(self, rollouts: list[EpisodeRollout]) -> None:
        with self._lock:
            for r in rollouts:
                self.transcripts.append(r.transcript)
            n = len(rollouts)
            if n:
                self.last_batch_stats = {
                    "rollout/win_rate": sum(r.stats.get("win", 0.0) for r in rollouts) / n,
                    "rollout/illegal_per_ep": sum(r.stats.get("num_illegal", 0) for r in rollouts) / n,
                    "rollout/repeat_per_ep": sum(r.stats.get("num_repeats", 0) for r in rollouts) / n,
                    "rollout/mean_turns": sum(r.stats.get("turns_used", 0) for r in rollouts) / n,
                    "rollout/budget_truncated_frac": sum(1 for r in rollouts if r.budget_truncated) / n,
                }

    def record_rewards(self, rewards: list[float], num_generations: int) -> None:
        groups = [
            rewards[i : i + num_generations] for i in range(0, len(rewards), num_generations)
        ]
        zero_var = sum(
            1 for grp in groups if len(grp) > 1 and statistics.pstdev(grp) < 1e-9
        )
        with self._lock:
            self.last_reward_stats = {
                "reward/mean": sum(rewards) / len(rewards) if rewards else 0.0,
                "reward/zero_variance_group_frac": zero_var / len(groups) if groups else 0.0,
            }

    def snapshot_metrics(self) -> dict:
        with self._lock:
            return {**self.last_batch_stats, **self.last_reward_stats}

    def sample_transcripts(self, k: int = 3) -> list[str]:
        with self._lock:
            items = list(self.transcripts)
        return items[-k:]


# ---------------------------------------------------------------- 答案取樣

def make_answer_sampler(answers: list, seed: int) -> Callable[[], Any]:
    """seeded 無限循環取樣：每輪 epoch 重洗。同一 prompt 槽位的一組 rollout 共用一次取樣。"""
    rng = random.Random(seed)
    pool: list = []

    def sample():
        nonlocal pool
        if not pool:
            pool = list(answers)
            rng.shuffle(pool)
        return pool.pop()

    return sample


# ---------------------------------------------------------------- TRL 轉接面

class SpikeValidationError(RuntimeError):
    """TRL 實驗性 API 與預期不符——這是 M2.1 spike 要當場解決的點。"""


def _trl_generate_fn(trainer, per_turn_max_tokens: int, temperature: float, top_p: float,
                     stop: tuple[str, ...]) -> GenerateFn:
    """把 GRPOTrainer 的 vLLM（colocate/server 皆可）包成 GenerateFn。

    【M2.1 spike 驗證點 A2】generate_rollout_completions 的簽名與取樣覆寫：
    研究（2026-07-10）指出 trl.experimental.openenv.generate_rollout_completions(trainer,
    prompts) 會沿用 trainer 取樣設定並在 server/colocate 兩模式可用；per-turn 覆寫
    （max_tokens/stop）若不被接受，退而直呼 trainer 的 vLLM engine。以下逐一嘗試、
    全滅則拋 SpikeValidationError 給 spike 記錄。
    """
    try:
        from trl.experimental.openenv import generate_rollout_completions  # type: ignore
    except ImportError as e:
        raise SpikeValidationError(
            f"無法 import trl.experimental.openenv.generate_rollout_completions：{e}；"
            "確認 trl 版本 pin（requirements-colab.txt）或改走 vLLM engine 直呼路徑"
        ) from e

    tok = trainer.processing_class

    def render(chats: list[list[dict]]) -> list[str]:
        return [
            tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in chats
        ]

    _MISSING = object()

    def to_turngen(item: Any) -> TurnGen:
        # 兼容 dict / dataclass 兩種回傳形態
        if isinstance(item, dict):
            get = lambda k, d=_MISSING: item.get(k, d)  # noqa: E731
            available = sorted(item.keys())
        else:
            get = lambda k, d=_MISSING: getattr(item, k, d)  # noqa: E731
            available = sorted(a for a in dir(item) if not a.startswith("_"))

        ids_raw = get("completion_ids")
        lps_raw = get("logprobs")
        # 欄位對不上就立刻報錯附真實可用欄位名——絕不能靜默塞空 list，
        # 那會讓 GRPO 拿零長度 completion 去算 loss，在 CUDA 底層炸出難懂的
        # index-out-of-bounds，而不是在這裡就講清楚問題出在哪（M2.1 spike 實錄）。
        if ids_raw is _MISSING or lps_raw is _MISSING:
            raise SpikeValidationError(
                "generate_rollout_completions 單筆回傳沒有預期的 completion_ids/logprobs 欄位"
                f"（型別={type(item).__name__}，實際可用欄位/屬性={available}）；"
                "對照這份清單改 to_turngen() 裡的欄位名"
            )
        ids = list(ids_raw)
        lps = list(lps_raw)
        text = get("text", None) or tok.decode(ids, skip_special_tokens=False)
        return TurnGen(token_ids=tuple(ids), logprobs=tuple(lps), text=text)

    def generate(chats: list[list[dict]]) -> list[TurnGen]:
        prompts_text = render(chats)
        attempts = (
            dict(max_tokens=per_turn_max_tokens, stop=list(stop),
                 temperature=temperature, top_p=top_p),
            dict(max_tokens=per_turn_max_tokens, stop=list(stop)),
            dict(),
        )
        last_err: Exception | None = None
        for kwargs in attempts:
            try:
                results = generate_rollout_completions(trainer, prompts_text, **kwargs)
                return [to_turngen(r) for r in results]
            except TypeError as e:  # 簽名不合 → 換下一組
                last_err = e
        raise SpikeValidationError(
            f"generate_rollout_completions 所有已知簽名皆失敗，最後錯誤：{last_err}；"
            "spike 時改測 trainer.llm / trainer.vllm_client 直呼路徑並回填此函數"
        )

    return generate


def make_rollout_func(
    game_factory: Callable[[Any], TurnBasedGame],
    answers: list,
    num_generations: int,
    per_turn_max_tokens: int,
    max_total_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop: tuple[str, ...] = ("</guess>",),
    seed: int = 42,
    buffers: RolloutBuffers | None = None,
    generate_fn_factory: Callable[..., GenerateFn] | None = None,
):
    """建 TRL rollout_func(prompts, trainer) -> dict。

    【M2.1 spike 實測修正】TRL 1.8 真實契約（trl/trainer/utils.py 的 RepeatSampler +
    grpo_trainer.py 的 sampler 建構參數 mini_repeat_count=num_generations 已讀原始碼確認，
    推翻了規格原先「prompts 未依 num_generations 重複」的研究假設）：
    - **prompts 已經是 TRL 用 RepeatSampler 展開好的完整批次**：每個底層 dataset row
      被連續重複 num_generations 次才送進來，len(prompts) 已等於 generation_batch_size
      （= num_generations 的整數倍）。本函數只能對每個收到的 prompt 產生「一局」，
      每 num_generations 個連續 prompt 共用「一個」抽樣答案——絕不能再乘一次
      num_generations，否則回傳筆數是 TRL 預期的 num_generations 倍，會在
      shuffle_sequence_dict 的 permute 階段索引越界（M2.1 spike 實際撞到的 CUDA crash）。
    - 必要鍵：prompt_ids / completion_ids / logprobs；其他鍵轉發給 reward functions。
    """
    sampler = make_answer_sampler(answers, seed=seed)
    gen_factory = generate_fn_factory or _trl_generate_fn

    def rollout_func(prompts: list, trainer) -> dict:
        if len(prompts) % num_generations != 0:
            raise SpikeValidationError(
                f"收到 {len(prompts)} 個 prompts，不是 num_generations={num_generations} 的整數倍"
                "——TRL 的 RepeatSampler 契約假設不成立，檢查 GRPOConfig 的 "
                "generation_batch_size / per_device_train_batch_size 設定"
            )
        games: list[TurnBasedGame] = []
        answers_used: list[Any] = []
        ans: Any = None
        for i in range(len(prompts)):
            if i % num_generations == 0:
                ans = sampler()  # 每 num_generations 個連續 prompt 共用一個答案
            games.append(game_factory(ans))
            answers_used.append(ans)

        generate = gen_factory(
            trainer,
            per_turn_max_tokens=per_turn_max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        rollouts = play_batch_episodes(
            generate,
            games,
            per_turn_max_tokens=per_turn_max_tokens,
            max_total_tokens=max_total_tokens,
        )
        if buffers is not None:
            buffers.record_rollouts(rollouts)

        tok = trainer.processing_class
        # 開局 messages 不含答案資訊，同組（甚至跨組）內容都相同，但仍逐一 tokenize
        # 以維持 len(prompt_ids) == len(games) == len(prompts) 的一一對應。
        prompt_ids: list[list[int]] = []
        for game in games:
            initial_text = tok.apply_chat_template(
                _initial_messages(game), tokenize=False, add_generation_prompt=True
            )
            ids = tok(initial_text, add_special_tokens=False)["input_ids"]
            prompt_ids.append(list(ids))

        out: dict[str, list] = {
            "prompt_ids": prompt_ids,
            "completion_ids": [r.completion_ids for r in rollouts],
            "logprobs": [r.logprobs for r in rollouts],
            "answer_used": [str(a) for a in answers_used],
        }
        stat_keys = sorted({k for r in rollouts for k in r.stats})
        for k in stat_keys:
            out[k] = [r.stats.get(k, 0.0) for r in rollouts]
        return out

    return rollout_func


def _initial_messages(game: TurnBasedGame) -> list[dict]:
    """取 episode 開局（無歷史）的 messages。games 的 build_messages 在無歷史時即是。"""
    msgs = game.build_messages()
    # 防衛：開局應該只有 system + 首個 user
    return msgs[:2]


# ---------------------------------------------------------------- reward 轉接

def make_trl_reward_fn(
    game: str,
    preset: str,
    num_generations: int,
    buffers: RolloutBuffers | None = None,
    reward_config: RewardConfig | None = None,
):
    """讀 rollout_func 轉發欄位 → 純獎勵函數。簽名符合 TRL reward_funcs 慣例。"""
    cfg = reward_config or RewardConfig()

    def wordle_stats_to_reward(i: int, kw: dict) -> float:
        stats = EpisodeStats(
            answer=kw.get("answer_used", ["?"] * (i + 1))[i],
            won=bool(kw["win"][i]),
            turns_used=int(kw["turns_used"][i]),
            num_illegal=int(kw["num_illegal"][i]),
            num_violation_turns=int(kw["num_violation_turns"][i]),
            num_repeats=int(kw["num_repeats"][i]),
            total_new_greens=int(kw["total_new_greens"][i]),
            total_new_presence=int(kw["total_new_presence"][i]),
            turns=[],
        )
        return episode_reward(stats, preset, cfg)

    def number_stats_to_reward(i: int, kw: dict) -> float:
        return number_guess_reward(
            {
                "win": kw["win"][i],
                "turns_used": int(kw["turns_used"][i]),
                "num_illegal": int(kw["num_illegal"][i]),
                "num_repeats": int(kw["num_repeats"][i]),
            }
        )

    per_episode = wordle_stats_to_reward if game == "wordle" else number_stats_to_reward

    def reward_fn(completions=None, **kwargs) -> list[float]:
        n = len(kwargs["win"])
        rewards = [float(per_episode(i, kwargs)) for i in range(n)]
        if buffers is not None:
            buffers.record_rewards(rewards, num_generations)
        return rewards

    reward_fn.__name__ = f"{game}_{preset}_reward"
    return reward_fn
