"""TrainerCallback 工廠：時間制 checkpoint、樣本傾印、metrics.jsonl、--max-hours 收尾。

transformers 只在工廠函數內 import——本機無 torch 環境仍可 import 本模組。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .rollout import RolloutBuffers


def build_callbacks(
    buffers: RolloutBuffers,
    samples_dir: Path,
    metrics_path: Path,
    checkpoint_minutes: int,
    sample_every_steps: int,
    max_hours: float | None = None,
    transcripts_per_dump: int = 3,
) -> list:
    from transformers import TrainerCallback

    class TimedCheckpointCallback(TrainerCallback):
        """時間制存檔（規格：每 30 分鐘上 Drive）——不用猜 sec/step。"""

        def __init__(self, minutes: int):
            self.interval = minutes * 60
            self.last = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            if time.monotonic() - self.last >= self.interval:
                control.should_save = True
                self.last = time.monotonic()
            return control

    class MaxHoursCallback(TrainerCallback):
        """牆鐘預算到 → 存檔 + 優雅停訓（把 push HF 的時間留在 Colab 額度內）。"""

        def __init__(self, hours: float | None):
            self.deadline = (time.monotonic() + hours * 3600) if hours else None

        def on_step_end(self, args, state, control, **kwargs):
            if self.deadline and time.monotonic() >= self.deadline:
                control.should_save = True
                control.should_training_stop = True
            return control

    class SampleDumpCallback(TrainerCallback):
        """每 N 步存 3 局完整 transcript → samples/step_{k}.md（肉眼查 reward hacking）。"""

        def __init__(self, every: int):
            self.every = every
            samples_dir.mkdir(parents=True, exist_ok=True)

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step > 0 and state.global_step % self.every == 0:
                transcripts = buffers.sample_transcripts(transcripts_per_dump)
                if transcripts:
                    path = samples_dir / f"step_{state.global_step:05d}.md"
                    path.write_text(
                        f"# step {state.global_step}\n\n" + "\n---\n\n".join(transcripts),
                        encoding="utf-8",
                    )
            return control

    class MetricsJsonlCallback(TrainerCallback):
        """trainer log + rollout/reward 彙整 → metrics.jsonl（監控 cell 畫曲線用）。"""

        def __init__(self):
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            record = {
                "step": state.global_step,
                "wallclock": time.time(),
                **(logs or {}),
                **buffers.snapshot_metrics(),
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return control

    return [
        TimedCheckpointCallback(checkpoint_minutes),
        MaxHoursCallback(max_hours),
        SampleDumpCallback(sample_every_steps),
        MetricsJsonlCallback(),
    ]
