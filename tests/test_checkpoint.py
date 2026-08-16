from datetime import date
from pathlib import Path

from src.checkpoint import (
    day_is_complete,
    restore_checkpoints,
    save_day_checkpoint,
)


def test_save_and_restore_roundtrip(tmp_path: Path):
    out = tmp_path / "run" / "learning_demo"
    ckpt = tmp_path / "checkpoints"
    out.mkdir(parents=True)
    day = date(2021, 4, 15)
    stem = f"BTCUSDT_{day.isoformat()}"
    (out / f"{stem}_ewma_state.json").write_text('{"expected_ticks": 20000}')
    (out / f"{stem}_summary.json").write_text('{"n_bars": 12}')
    (out / f"{stem}_labels.csv").write_text("y_meta\n1\n")
    copied = save_day_checkpoint(out, ckpt, "BTCUSDT", day)
    assert f"{stem}_ewma_state.json" in copied
    assert f"{stem}_labels.csv" in copied
    assert (ckpt / "learning_demo" / "LAST.json").exists()

    restored_out = tmp_path / "run2" / "learning_demo"
    n = restore_checkpoints(ckpt, restored_out, run_id="learning_demo")
    assert n >= 3
    assert day_is_complete(restored_out, "BTCUSDT", day, bars_only=False)
    assert not day_is_complete(restored_out, "BTCUSDT", date(2021, 4, 16), bars_only=False)


def test_warmup_complete_without_labels(tmp_path: Path):
    folder = tmp_path / "out"
    folder.mkdir()
    day = date(2017, 8, 17)
    (folder / f"BTCUSDT_{day}_ewma_state.json").write_text("{}")
    assert day_is_complete(folder, "BTCUSDT", day, bars_only=True)
    assert not day_is_complete(folder, "BTCUSDT", day, bars_only=False)
