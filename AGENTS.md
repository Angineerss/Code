# AGENTS.md

## Cursor Cloud specific instructions

This is a pure-Python research pipeline (Binance aggTrades → dollar bars →
triple-barrier meta labels → CPCV). Default clock is `bar_type=dollar`;
`dollar_imbalance` is the original sampler kept as a control. There is no
server, web UI, or database; the "application" is the CLI in `src/` plus
the scripts in `scripts/`.

### Environment / tooling

- Use `python3` (Python 3.12). There is no `python` alias on this VM.
- Dependencies are installed with `pip install -r requirements.txt` (the startup update
  script already does this). They land in `~/.local`, which is why console scripts like
  `pytest`/`pygmentize` are not on `PATH` — always invoke tools as modules, e.g.
  `python3 -m pytest`, not bare `pytest`.
- `requirements.txt` is unpinned (`>=`), so latest `numpy`/`pandas`/`scikit-learn` get
  installed. The suite passed on pandas 3.x / scikit-learn 1.9; no compatibility pins
  are needed as of this writing.

### Tests

- Run the full suite with `python3 -m pytest` (config in `pytest.ini`: `pythonpath=.`,
  `testpaths=tests`). Tests use synthetic ticks (`tests/helpers.py`) and require **no
  network** — they are fully self-contained.

### Lint

- No linter/formatter is configured (no ruff/flake8/black/mypy). "Lint" for this repo is
  effectively just the test suite.

### Running the application (network required)

- `python3 -m src --symbol BTCUSDT --date YYYY-MM-DD` runs the whole pipeline for one UTC
  day. Unlike the tests, this **downloads real data from `data.binance.vision` and
  `api.binance.com`**, so it needs outbound network access. A single BTCUSDT day is a
  ~18 MB aggTrades zip plus small 1d-kline zips; a run takes ~10-15s.
- Outputs (bars/labels/ewma_state/summary CSVs+JSON and cached zips) are written under
  `data/`, which is **git-ignored** — do not commit it. Downloaded zips are cached and
  reused on re-runs.
- OOS calendar days (`2025-01-01` onward) are **rejected by default**; the code refuses
  them unless you pass `--allow-oos`. Use in-sample dates (e.g. `2024-01-15`) for quick
  smoke tests. `EWMA` state is chained day-to-day via
  `data/{SYMBOL}_{date-1}_ewma_state.json`.
- The bulk `scripts/` (`download_aggtrades_archive.py`, `run_learning_range.py`, etc.)
  download the full multi-year Vision archive (58 GB+) — do not run them for smoke tests.
