"""
PageOracle Dev Quality Gates safe autofix runner.
Applies safe automatic formatting and lint fixes.
"""

from __future__ import annotations

import subprocess
import sys

FIX_COMMANDS: list[list[str]] = [
    ["uvx", "ruff", "format", "."],
    ["uvx", "ruff", "check", "--fix", "."],
    ["uvx", "rumdl", "fmt", "."],
]


def main() -> int:
    for cmd in FIX_COMMANDS:
        cmd_str = " ".join(cmd)
        print(f"\n[fix] Running: {cmd_str}")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[fix] FAILED: {cmd_str} (exit code {res.returncode})", file=sys.stderr)
            return res.returncode
        print(f"[fix] DONE: {cmd_str}")

    print("\n[fix] Safe autofixes applied successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
