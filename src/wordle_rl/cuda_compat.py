"""Colab CUDA 環境相容修正——所有會 import vllm 的入口共用。

【事故記錄】這個修正原本只放在 train.py（M2.1 spike 診斷實錄），結果 v3 正式訓練
跑完、權重也保全了，eval/run_eval.py 一啟動就炸出一模一樣的
`ImportError: libcudart.so.13`——因為評測入口從來沒真正在 Colab 上執行過
（前幾輪都死在更早的階段），身上沒有這個修正。教訓：環境修正必須放在
「靠近依賴」的共用模組，而不是複製到每個入口。現在 train.py 與
backends.VLLMBackend 都呼叫這裡。

問題本體：vllm 0.23.0 的編譯擴充 vllm._C 連到 libcudart.so.13，pip 裝的
`nvidia-cuda-runtime`（無 -cu12 尾綴＝cu13）有提供該 .so，但不在動態連結器的
預設搜尋路徑上。

第一版修法（只設 os.environ["LD_LIBRARY_PATH"]）在真實 Colab 環境驗證失敗：
LD_LIBRARY_PATH 只在 process 啟動當下被動態連結器讀取一次，process 跑起來後改它
對「目前這個 process」後續的 import/dlopen 沒有作用。真正有效的做法：用
ctypes.CDLL(..., RTLD_GLOBAL) 把 .so 直接預先載入目前 process——之後任何 import
觸發的 dlopen 找同名庫會直接命中已載入的這份。同時仍設 LD_LIBRARY_PATH，讓
vllm 可能 fork/spawn 的子行程（全新 exec 啟動）也能讀到路徑。

找不到 .so 時整個函式是無害 no-op（例如本機 Windows / 非 Colab Linux）。
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from pathlib import Path

_CUDA13_RUNTIME_RELATIVE_PATH = Path("nvidia/cu13/lib/libcudart.so.13")

_applied = False


def _python_package_roots() -> tuple[Path, ...]:
    """Return the finite, deterministic set of active Python import roots."""
    roots: set[Path] = set()
    for entry in sys.path:
        if not entry:
            continue
        try:
            roots.add(Path(entry).absolute())
        except (OSError, TypeError):
            continue
    return tuple(sorted(roots, key=os.fspath))


def _discover_cuda13_runtime_dirs(
    package_roots: Iterable[str | os.PathLike[str]] | None = None,
) -> tuple[Path, ...]:
    """Check one exact CUDA runtime path below each bounded Python root."""
    roots = _python_package_roots() if package_roots is None else package_roots
    directories: set[Path] = set()
    for root in roots:
        try:
            runtime = Path(root) / _CUDA13_RUNTIME_RELATIVE_PATH
            if runtime.is_file():
                directories.add(runtime.parent)
        except (OSError, TypeError):
            continue
    return tuple(sorted(directories, key=os.fspath))


def fix_missing_cuda13_runtime_ld_path() -> None:
    global _applied
    if _applied:
        return
    _applied = True

    if sys.platform != "linux":
        return

    dirs = _discover_cuda13_runtime_dirs()
    if not dirs:
        return
    library_paths = [os.fspath(directory) for directory in dirs]
    if existing := os.environ.get("LD_LIBRARY_PATH"):
        library_paths.append(existing)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
    import ctypes

    for directory in dirs:
        runtime = directory / _CUDA13_RUNTIME_RELATIVE_PATH.name
        try:
            ctypes.CDLL(os.fspath(runtime), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
