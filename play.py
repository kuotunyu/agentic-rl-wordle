"""Backward-compatible wrapper for the installed ``wordle-rl`` CLI."""

from wordle_rl.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
