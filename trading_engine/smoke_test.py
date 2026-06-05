"""
Minimal end-to-end smoke test with synthetic features_d.

Run as: python trading_engine/smoke_test.py
Exits with code 0 on success, non-zero on failure.
"""
import math
import sys
import os
import logging

# Ensure project root is on sys.path and set as cwd so outputs/ lands there
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(_PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)  # suppress INFO noise during test


def _make_synthetic_features_d() -> pd.DataFrame:
    """Build a minimal features_d DataFrame with known buy/sell events.

    Price rises for 40 days then drops sharply. sl_w is fixed at a constant
    level (base_close - sl_offset) so the declining price eventually crosses
    it and triggers a sell.
    """
    dates = pd.date_range("2020-01-02", periods=60, freq="B")  # 60 business days

    rows = []
    for sym, base_close, sl_offset in [("ALPHA", 100.0, 10.0), ("BETA", 200.0, 20.0)]:
        # Fixed stop-loss: will be crossed when price drops below base_close
        fixed_sl = base_close - sl_offset  # e.g., ALPHA: 90, BETA: 180

        for i, d in enumerate(dates):
            # Uptrend for 40 days then sharp decline
            if i < 40:
                close = base_close + i * 0.5          # rises to base+20
            else:
                close = base_close + 20 - (i - 40) * 2.5  # falls ~2.5/day

            # sl_w is fixed — price will cross it around day 48
            sl_w = fixed_sl

            # Validate invariant: sl_w must be < close at buy time
            buy = 1 if i in (5, 15) and sl_w < close else 0

            # ema21 slightly below close
            ema21 = close - 2.0

            rows.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "close": round(close, 2),
                    "buy": buy,
                    "ema21": round(ema21, 4),
                    "sl_w": round(sl_w, 4),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    from trading_engine.engine import run_backtest

    print("Building synthetic features_d...")
    df = _make_synthetic_features_d()

    print(f"  Shape: {df.shape}, buy==1 rows: {(df['buy']==1).sum()}")

    # Validate no sl_w >= close on buy==1 rows
    buy_rows = df[df["buy"] == 1]
    violations = buy_rows[buy_rows["sl_w"] >= buy_rows["close"]]
    assert len(violations) == 0, f"Synthetic data has {len(violations)} sl_w >= close violations"

    print("Running backtest...")
    results = run_backtest(df)

    # ── Structural checks ─────────────────────────────────────────────────────
    assert "trade_log" in results
    assert "daily_signals" in results
    assert "portfolio_daily" in results
    assert "trade_summary" in results

    tl = results["trade_log"]
    ds = results["daily_signals"]
    pd_ = results["portfolio_daily"]
    ts = results["trade_summary"]

    # trade_log must have expected columns
    required_tl_cols = {
        "event_id", "trade_count", "ticker", "event_type", "date",
        "execution_price", "units_transacted", "units_held_after",
        "avg_buy_price", "stop_loss_price", "stop_loss_pct",
        "position_size_used", "portfolio_value", "cash_after",
        "realized_gain_abs", "realized_gain_pct",
    }
    missing_tl = required_tl_cols - set(tl.columns)
    assert not missing_tl, f"trade_log missing columns: {missing_tl}"

    # daily_signals must have expected columns
    required_ds_cols = {
        "date", "ticker", "close", "sl_w", "stop_loss_pct",
        "buy_signal", "sell_signal", "in_portfolio", "units_held",
        "avg_buy_price", "position_size_estimate", "units_purchaseable",
        "portfolio_value_bod",
    }
    missing_ds = required_ds_cols - set(ds.columns)
    assert not missing_ds, f"daily_signals missing columns: {missing_ds}"

    # portfolio_daily must have expected columns
    required_pd_cols = {
        "date", "portfolio_value", "cash", "invested_value",
        "daily_return", "cumulative_return",
    }
    missing_pd = required_pd_cols - set(pd_.columns)
    assert not missing_pd, f"portfolio_daily missing columns: {missing_pd}"

    # trade_summary must have expected columns
    required_ts_cols = {
        "ticker", "trade_count", "buy_date", "sell_date",
        "avg_buy_price", "sell_price", "units", "cost_basis",
        "proceeds", "realized_gain_abs", "realized_gain_pct", "holding_days",
    }
    missing_ts = required_ts_cols - set(ts.columns)
    assert not missing_ts, f"trade_summary missing columns: {missing_ts}"

    # Row count checks
    assert len(ds) == len(df), (
        f"daily_signals rows ({len(ds)}) != input rows ({len(df)})"
    )
    assert len(pd_) == df["date"].nunique(), (
        f"portfolio_daily rows ({len(pd_)}) != unique dates ({df['date'].nunique()})"
    )

    # At least some buy events should have fired
    buys = tl[tl["event_type"] == "BUY"]
    assert len(buys) > 0, "No BUY events logged — check buy trigger logic"
    print(f"  BUY events: {len(buys)}")

    # At least one sell event should have fired (sharp decline after day 40)
    sells = tl[tl["event_type"] == "SELL"]
    assert len(sells) > 0, "No SELL events logged — check stop-loss trigger logic"
    print(f"  SELL events: {len(sells)}")

    # Portfolio value must be positive throughout
    assert (pd_["portfolio_value"] > 0).all(), "Portfolio value went to zero or negative"

    # Cash must never go negative
    assert (pd_["cash"] >= 0).all(), "Cash went negative — position sizing error"

    # Sell events must have NaN avg_buy_price
    sell_rows = tl[tl["event_type"] == "SELL"]
    assert sell_rows["avg_buy_price"].isna().all(), (
        "SELL events should have NaN avg_buy_price"
    )

    # Sell events must have non-NaN realized gain
    assert sell_rows["realized_gain_abs"].notna().all(), (
        "SELL events should have realized_gain_abs populated"
    )

    # Parquet outputs must exist
    from pathlib import Path
    for name in ["trade_log", "daily_signals", "portfolio_daily", "trade_summary"]:
        p = Path("outputs") / f"{name}.parquet"
        assert p.exists(), f"Missing output file: {p}"

    # Summary stats must be computable
    stats = results["summary_stats"]
    assert "total_trades" in stats
    assert stats["total_trades"] == len(ts), (
        f"total_trades mismatch: stats={stats['total_trades']}, len(ts)={len(ts)}"
    )

    print("\nAll assertions passed.")
    print(f"  Total trades (closed): {stats['total_trades']}")
    print(f"  Win rate:              {stats.get('win_rate', math.nan):.1%}")
    print(f"  Cumulative return:     {stats.get('portfolio_cumulative_return_pct', math.nan):.2f}%")
    print(f"  Sharpe ratio:          {stats.get('sharpe_ratio', math.nan):.3f}")
    print(f"  Max drawdown:          {stats.get('max_drawdown_pct', math.nan):.2f}%")


if __name__ == "__main__":
    try:
        main()
        print("\nSmoke test PASSED.")
        sys.exit(0)
    except Exception as exc:
        print(f"\nSmoke test FAILED: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
