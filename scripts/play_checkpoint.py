"""Compatibility entrypoint for playing the original CarryBox checkpoint."""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("play_carrybox.py")), run_name="__main__")
