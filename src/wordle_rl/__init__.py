"""agentic-rl-wordle：多輪 GRPO 訓練 Qwen2.5-1.5B-Instruct 玩 Wordle。

核心模組（words/env/knowledge/protocol/episode/metrics/rewards）零重依賴，
Windows CPU 上 pytest 秒級可跑；torch/transformers/vllm/trl 只在推理與訓練模組載入。
"""

__version__ = "1.0.0"
