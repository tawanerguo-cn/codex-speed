"""Run the bundled Codex speed CLI without installing project dependencies."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    skill_root = Path(__file__).resolve().parent.parent
    vendor = skill_root / "_vendor"
    if not vendor.is_dir():
        raise SystemExit("The bundled _vendor/token_pulse package is missing; rebuild the plugin.")
    sys.path.insert(0, str(vendor))
    runpy.run_module("token_pulse.speed", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
