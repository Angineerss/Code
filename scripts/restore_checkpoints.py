#!/usr/bin/env python3
"""Copy tracked checkpoints back into an ephemeral data/runs directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint import restore_checkpoints


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", default="results/checkpoints")
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--out-dir",
        default=None,
        help="Default: data/runs/<run-id>",
    )
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir or f"data/runs/{args.run_id}")
    n = restore_checkpoints(Path(args.checkpoint_dir), out_dir, run_id=args.run_id)
    print(json.dumps({"restored": n, "out_dir": str(out_dir)}, indent=2))
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
