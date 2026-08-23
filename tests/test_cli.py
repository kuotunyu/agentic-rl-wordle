import locale
import os
import subprocess
import sys


def test_module_help_works_outside_repository_without_optional_dependencies(tmp_path):
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-m", "wordle_rl", "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding=locale.getpreferredencoding(False),
        errors="replace",
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--answer" in result.stdout
    assert "--adapter" in result.stdout
