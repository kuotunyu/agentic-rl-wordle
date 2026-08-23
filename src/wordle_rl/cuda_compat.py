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

import glob
import os

_applied = False


def fix_missing_cuda13_runtime_ld_path() -> None:
    global _applied
    if _applied:
        return
    _applied = True

    dirs = {
        os.path.dirname(p)
        for p in glob.glob("/usr/**/nvidia/cu13/lib/libcudart.so.13", recursive=True)
    }
    if not dirs:
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    import ctypes

    for d in dirs:
        so_path = os.path.join(d, "libcudart.so.13")
        if os.path.exists(so_path):
            try:
                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
