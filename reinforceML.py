from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    command = [
        "uv",
        "run",
        "--project",
        str(PROJECT_ROOT),
        "fruitmatch-ml",
        "serve",
    ]

    return subprocess.call(command, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())