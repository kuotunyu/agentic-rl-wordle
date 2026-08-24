"""Regression coverage for bounded CUDA 13 runtime discovery."""

from __future__ import annotations

import ctypes
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

from wordle_rl import cuda_compat

CUDA_RUNTIME_RELATIVE_PATH = Path("nvidia/cu13/lib/libcudart.so.13")


def _runtime_file(root: Path) -> Path:
    runtime = root / CUDA_RUNTIME_RELATIVE_PATH
    runtime.parent.mkdir(parents=True)
    runtime.touch()
    return runtime


def _configure_linux_roots(monkeypatch: pytest.MonkeyPatch, *roots: Path) -> None:
    monkeypatch.setattr(cuda_compat.sys, "platform", "linux")
    monkeypatch.setattr(cuda_compat, "_python_package_roots", lambda: tuple(roots))


def test_source_forbids_unbounded_recursive_filesystem_searches():
    source = inspect.getsource(cuda_compat)
    forbidden = (
        "recursive=True",
        "/**",
        ".rglob(",
        'os.walk("/")',
        "os.walk('/')",
        'os.walk("/usr")',
        "os.walk('/usr')",
    )

    assert not [pattern for pattern in forbidden if pattern in source]


def test_python_package_roots_are_finite_deduplicated_and_deterministic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    first = tmp_path / "z-package-root"
    second = tmp_path / "a-package-root"
    monkeypatch.setattr(cuda_compat.sys, "path", [str(first), "", str(second), str(first)])

    roots = cuda_compat._python_package_roots()

    assert roots == tuple(sorted({first.absolute(), second.absolute()}, key=os.fspath))


def test_discovery_uses_only_exact_relative_paths_within_explicit_roots(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_runtime = _runtime_file(first)
    second_runtime = _runtime_file(second)
    _runtime_file(tmp_path / "outside" / "nested")
    near_miss = first / "nested" / CUDA_RUNTIME_RELATIVE_PATH
    near_miss.parent.mkdir(parents=True)
    near_miss.touch()

    directories = cuda_compat._discover_cuda13_runtime_dirs(
        (second, first, second, tmp_path / "empty")
    )

    assert directories == tuple(
        sorted({first_runtime.parent, second_runtime.parent}, key=os.fspath)
    )


def test_runtime_discovery_finds_bounded_python_package_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runtime = _runtime_file(tmp_path)
    _configure_linux_roots(monkeypatch, tmp_path)

    assert cuda_compat._discover_cuda13_runtime_dirs() == (runtime.parent,)


def test_missing_runtime_is_a_noop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _configure_linux_roots(monkeypatch, tmp_path)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(cuda_compat, "_applied", False)
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(ctypes, "CDLL", lambda path, mode: calls.append((path, mode)))

    cuda_compat.fix_missing_cuda13_runtime_ld_path()

    assert "LD_LIBRARY_PATH" not in os.environ
    assert calls == []


def test_non_linux_runtime_discovery_is_a_noop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _runtime_file(tmp_path)
    monkeypatch.setattr(cuda_compat.sys, "platform", "win32")
    monkeypatch.setattr(cuda_compat, "_python_package_roots", lambda: (tmp_path,))
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(cuda_compat, "_applied", False)
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(ctypes, "CDLL", lambda path, mode: calls.append((path, mode)))

    cuda_compat.fix_missing_cuda13_runtime_ld_path()

    assert "LD_LIBRARY_PATH" not in os.environ
    assert calls == []


def test_runtime_updates_ld_library_path_and_preloads_globally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runtime = _runtime_file(tmp_path)
    _configure_linux_roots(monkeypatch, tmp_path)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")
    monkeypatch.setattr(cuda_compat, "_applied", False)
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(ctypes, "CDLL", lambda path, mode: calls.append((path, mode)))

    cuda_compat.fix_missing_cuda13_runtime_ld_path()

    assert os.environ["LD_LIBRARY_PATH"] == f"{runtime.parent}{os.pathsep}/existing/lib"
    assert calls == [(os.fspath(runtime), ctypes.RTLD_GLOBAL)]


def test_runtime_fix_is_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    runtime = _runtime_file(tmp_path)
    _configure_linux_roots(monkeypatch, tmp_path)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(cuda_compat, "_applied", False)
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(ctypes, "CDLL", lambda path, mode: calls.append((path, mode)))

    cuda_compat.fix_missing_cuda13_runtime_ld_path()
    cuda_compat.fix_missing_cuda13_runtime_ld_path()

    assert os.environ["LD_LIBRARY_PATH"] == os.fspath(runtime.parent)
    assert calls == [(os.fspath(runtime), ctypes.RTLD_GLOBAL)]


def test_preload_oserror_does_not_break_import_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runtime = _runtime_file(tmp_path)
    _configure_linux_roots(monkeypatch, tmp_path)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(cuda_compat, "_applied", False)

    def fail_preload(path: object, mode: object) -> None:
        raise OSError(f"cannot preload {path} with {mode}")

    monkeypatch.setattr(ctypes, "CDLL", fail_preload)

    cuda_compat.fix_missing_cuda13_runtime_ld_path()

    assert os.environ["LD_LIBRARY_PATH"] == os.fspath(runtime.parent)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux filesystem regression")
def test_linux_train_import_finishes_with_bounded_timeout(tmp_path: Path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    process = subprocess.Popen(
        [sys.executable, "-c", "import wordle_rl.train; print('import-ok')"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=3)
        pytest.fail(
            "wordle_rl.train import exceeded 8 seconds; child was killed and reaped\n"
            f"stdout={stdout!r}\nstderr={stderr!r}"
        )

    assert process.returncode == 0, stderr
    assert stdout.strip() == "import-ok"
