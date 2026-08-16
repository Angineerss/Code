"""Durable checkpoints so cloud disks can vanish without losing the run.

``data/`` is ephemeral (gitignored, cloud VM disk). Binance Vision still has
the ticks. This module copies the *computed* day artifacts into
``results/checkpoints/<run_id>/`` (tracked) and can ``git push`` them.

Resume: restore JSON/CSV back into ``data/runs/...``, then ``--skip-existing``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import date
from pathlib import Path

CHECKPOINT_SUFFIXES = (
    "_ewma_state.json",
    "_summary.json",
    "_labels.csv",
    "_events.csv",
    "_bars.csv",
)

MANIFEST_NAMES = (
    "learning_manifest.json",
    "PROGRESS.json",
)


def run_id_from_out_dir(out_dir: Path) -> str:
    return out_dir.name


def checkpoint_run_dir(checkpoint_root: Path, run_id: str) -> Path:
    return checkpoint_root / run_id


def day_stem(symbol: str, day: date) -> str:
    return f"{symbol}_{day.isoformat()}"


def day_is_complete(
    folder: Path,
    symbol: str,
    day: date,
    *,
    bars_only: bool,
) -> bool:
    """A day is done if EWMA exists, and labels exist unless bars-only."""
    stem = day_stem(symbol, day)
    if not (folder / f"{stem}_ewma_state.json").exists():
        return False
    if bars_only:
        return True
    return (folder / f"{stem}_labels.csv").exists()


def copy_if_exists(src: Path, dest: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def save_day_checkpoint(
    out_dir: Path,
    checkpoint_root: Path,
    symbol: str,
    day: date,
    *,
    run_id: str | None = None,
    start: date | None = None,
    end: date | None = None,
) -> list[str]:
    """Copy one day's artifacts plus progress JSON into the tracked checkpoint dir."""
    run_id = run_id or run_id_from_out_dir(out_dir)
    dest_dir = checkpoint_run_dir(checkpoint_root, run_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = day_stem(symbol, day)
    copied: list[str] = []
    for suffix in CHECKPOINT_SUFFIXES:
        name = f"{stem}{suffix}"
        if copy_if_exists(out_dir / name, dest_dir / name):
            copied.append(name)
    for extra in MANIFEST_NAMES:
        if copy_if_exists(out_dir / extra, dest_dir / extra):
            copied.append(extra)
    if start is not None and end is not None:
        progress_name = f"{symbol}_{start.isoformat()}_{end.isoformat()}_progress.json"
        if copy_if_exists(out_dir / progress_name, dest_dir / progress_name):
            copied.append(progress_name)
        if copy_if_exists(out_dir / progress_name, dest_dir / "PROGRESS.json"):
            copied.append("PROGRESS.json")
    pointer = {
        "run_id": run_id,
        "symbol": symbol,
        "last_day": day.isoformat(),
        "out_dir": str(out_dir),
        "copied": copied,
    }
    (dest_dir / "LAST.json").write_text(json.dumps(pointer, indent=2))
    copied.append("LAST.json")
    return copied


def restore_checkpoints(
    checkpoint_root: Path,
    out_dir: Path,
    *,
    run_id: str | None = None,
) -> int:
    """Copy checkpoint files back onto the ephemeral ``data/`` run dir."""
    run_id = run_id or run_id_from_out_dir(out_dir)
    src_dir = checkpoint_run_dir(checkpoint_root, run_id)
    if not src_dir.is_dir():
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in src_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, out_dir / path.name)
            n += 1
    return n


def push_checkpoints(repo: Path, checkpoint_root: Path, message: str) -> str:
    """Commit and push checkpoint files. Returns a status string; never raises."""
    rel = os.path.relpath(checkpoint_root, repo)
    try:
        subprocess.run(["git", "add", "--", rel], cwd=repo, check=True, capture_output=True)
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", rel],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        if not status.stdout.strip():
            return "clean"
        subprocess.run(["git", "commit", "-m", message], cwd=repo, check=True, capture_output=True)
        push = subprocess.run(
            ["git", "push"],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        if push.returncode != 0:
            return f"commit_ok_push_failed: {push.stderr.strip() or push.stdout.strip()}"
        return "pushed"
    except (subprocess.CalledProcessError, OSError) as exc:
        return f"failed: {exc}"
