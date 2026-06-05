# Assumptions & Confirmed Schema

## Confirmed `features_d` Schema

Inspected from `data/features/features_d.csv` and `src/engineering.py`.

| Column   | Dtype in CSV | Notes |
|----------|-------------|-------|
| `date`   | object (string, e.g. "2018-01-30") | Converted to `datetime64` at backtest start |
| `symbol` | object (str) | Ticker / security identifier |
| `close`  | float64 | Day's closing price — used as execution price |
| `buy`    | int64 (0/1) | Buy signal produced by `compute_buy_flag()` in engineering.py |
| `ema21`  | float64 | 21-period EMA of close; may be NaN for early rows |
| `sl_w`   | float64 | Weekly-mapped stop-loss (from `compute_features_w` `sl` column, mapped to daily via `map_weekly_to_daily`); may be NaN for early rows |

All 6 required columns are present. In the actual data, 2 943 rows have `buy == 1` and none of them have `sl_w >= close` — validation passes cleanly.

---

## Design Assumptions

**A1 — Portfolio value is frozen at BOD for the entire trading day.**
`portfolio_value` is computed once at the start of each date (cash + all open positions at that day's close) and used for all allocation and position-sizing calculations for that day. `cash` is updated in real-time as trades execute within the day, but `portfolio_value` stays fixed at the BOD snapshot. This is the standard approach for daily-bar backtesting.

**A2 — Within a date, tickers are processed in alphabetical order.**
The DataFrame is sorted by `['date', 'symbol']`, so within a date tickers are encountered A→Z. Cash from intra-day sells is immediately available for subsequent buys within the same date.

**A3 — Sell is evaluated before buy for each ticker on each date.**
If `current_close <= stop_loss_price` (sell trigger) AND `buy == 1` simultaneously for a ticker already in portfolio, the sell executes first and no buy is placed for that ticker on that date. In practice the validation constraint (`sl_w < close` for `buy == 1`) means a simultaneous sell + valid buy is impossible.

**A4 — For add-on buy (Case 2), NaN ema21 is treated as False.**
Python float comparison `current_close > float('nan')` returns `False`, so a NaN `ema21` falls through to the `current_close > last_buy_price` condition only.

**A5 — `buy_date` in `trade_summary` is the first buy date for a position.**
For positions with multiple add-on buys, `buy_date` is the date of the initial buy. `avg_buy_price` reflects the weighted average across all add-on buys at the time of sell.

**A6 — `avg_buy_price` is logged as NaN in SELL events.**
Per the schema: "Weighted avg; NaN after full sell." The avg_buy_price *before* sell is still used to compute `realized_gain_abs` and `realized_gain_pct`.

**A7 — Symbols absent from today's data retain their last `allocated_value`.**
If a symbol has an open position but no row in today's date group (e.g., early history), its contribution to `portfolio_value` is its last known `allocated_value`. Stop-loss checks are skipped for absent symbols.

**A8 — Units are always whole numbers (integer floor).**
All position sizing uses `math.floor(...)` to ensure integer unit counts.

**A9 — Outputs directory is relative to the working directory.**
`outputs/` and `outputs/plots/` are created relative to wherever the script is run. Intended to be run from the project root.

**A10 — Cash cap applied after headroom cap.**
After `compute_position_size` caps units by allocation headroom, the engine applies a further cap: `units_to_buy = min(units_to_buy, floor(cash / current_close))`. This prevents buying more than current cash allows even when headroom is large.
