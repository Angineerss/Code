"""Download Binance Vision BTCUSDT aggTrades from listing through the latest day."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PipelineConfig
from src.download import download_aggtrades_archive, latest_available_day


def main(argv: list[str] | None = None) -> int:
    config = PipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=config.symbol)
    parser.add_argument("--data-dir", default="data/aggtrades")
    parser.add_argument(
        "--start",
        default=config.archive_start.isoformat(),
        help="UTC start day YYYY-MM-DD (default = BTCUSDT Vision archive start)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="UTC end day YYYY-MM-DD (default = latest published Vision day)",
    )
    args = parser.parse_args(argv)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else latest_available_day(args.symbol)
    summary = download_aggtrades_archive(
        args.symbol.upper(),
        Path(args.data_dir),
        start,
        end,
    )
    print(json.dumps(summary, indent=2))
    dest = Path(args.data_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "download_manifest.json").write_text(json.dumps(summary, indent=2))
    return 0 if not summary["missing_months"] and not summary["missing_days"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
