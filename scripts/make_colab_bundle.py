"""打包原始碼成 wordle_rl_bundle.zip（repo 不上 GitHub → Colab 用 zip 取碼）。

用法：python scripts/make_colab_bundle.py
產出 repo 根目錄的 wordle_rl_bundle.zip（gitignored），手動上傳到
Google Drive 的 MyDrive/agentic-rl-wordle/ 供 wordle_grpo_colab_train.ipynb 解壓。
"""

from __future__ import annotations

import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "wordle_rl_bundle.zip"

INCLUDE = [
    "pyproject.toml",
    "requirements.txt",
    "requirements-colab.txt",
    "PLAN.md",
    "play.py",
    "src",
    "scripts",
    "baselines",
    "eval",
    "tests",
    "docs",
]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".git", "data", "runs", "results", "samples"}


def main() -> int:
    files: list[Path] = []
    for item in INCLUDE:
        p = REPO / item
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not (set(f.parts) & EXCLUDE_PARTS) and f.suffix != ".pyc":
                    files.append(f)
    # Fixed metadata makes the archive byte-for-byte reproducible across
    # machines. The full-463 notebook verifies this SHA256 so an older source
    # bundle cannot be uploaded by accident.
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            info = zipfile.ZipInfo(
                f.relative_to(REPO).as_posix(),
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, f.read_bytes())
    print(f"OK: {OUT.name}（{len(files)} 檔，{OUT.stat().st_size / 1024:.0f} KB）")
    print("→ 上傳到 Google Drive：MyDrive/agentic-rl-wordle/wordle_rl_bundle.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
