"""
PageOracle Dev Quality Gates aggregate check runner.
Runs the complete suite of quality gates in order.
"""

from __future__ import annotations

import subprocess
import sys

COMMANDS: list[list[str]] = [
    ["uvx", "ruff", "format", "--check", "."],
    ["uvx", "ruff", "check", "."],
    ["uvx", "ty", "check", "."],
    ["uvx", "deptry", "."],
    ["uvx", "typos"],
    ["uvx", "rumdl", "check", "."],
    ["uvx", "agnix", "."],
    ["uvx", "jactionlint", "--shellcheck="],
]


def main() -> int:
    failed = False
    for cmd in COMMANDS:
        cmd_str = " ".join(cmd)
        print(f"\n[gate] Running: {cmd_str}")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[gate] FAILED: {cmd_str} (exit code {res.returncode})", file=sys.stderr)
            failed = True
            break
        print(f"[gate] PASSED: {cmd_str}")

    if failed:
        print("\n[gate] Quality gate checks failed!", file=sys.stderr)
        return 1

    print("\n[gate] All quality gate checks passed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
