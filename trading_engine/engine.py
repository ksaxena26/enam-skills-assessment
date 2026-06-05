"""
Main backtest loop. Entry point: run_backtest(features_d) -> dict of DataFrames.
Run as a script to execute against the saved features_d.csv.
"""
import logging
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd

from trading_engine.config import (
    EXECUTION_PRICE_COL,
    INITIAL_CAPITAL,
    MAX_RISK_PER_TRADE,
    MAX_SECURITY_ALLOC,
)
from trading_engine.position_manager import (
    PositionState,
    compute_position_size,
    reset_position,
    update_position_on_buy,
)
from trading_engine.trade_logger import TradeLogger
from trading_engine import analytics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("outputs")
REQUIRED_COLS = {"date", "symbol", "close", "buy", "ema21", "sl_w"}


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_input(df: pd.DataFrame) -> None:
    """Assert required columns present; warn (don't raise) on sl_w >= close for buy==1."""
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"features_d is missing required columns: {missing}")

    buy_mask = df["buy"] == 1
    violations = df.loc[
        buy_mask
        & df["sl_w"].notna()
        & df["close"].notna()
        & (df["sl_w"] >= df["close"])
    ]
    for _, row in violations.iterrows():
        logger.warning(
            "Validation: sl_w (%.4f) >= close (%.4f) for %s on %s",
            row["sl_w"], row["close"], row["symbol"], row["date"],
        )


# ── Portfolio valuation helper ─────────────────────────────────────────────────

def _portfolio_value(
    cash: float,
    positions: Dict[str, PositionState],
    today_closes: Dict[str, float],
) -> float:
    """Cash + mark-to-market value of all open positions."""
    invested = 0.0
    for sym, state in positions.items():
        if state.units_held == 0:
            continue
        if sym in today_closes:
            invested += state.units_held * today_closes[sym]
        else:
            invested += state.allocated_value  # stale but best available
    return cash + invested


# ── Output serialisation ──────────────────────────────────────────────────────

def _write_outputs(
    trade_log: pd.DataFrame,
    daily_signals: pd.DataFrame,
    portfolio_daily: pd.DataFrame,
    trade_summary: pd.DataFrame,
) -> None:
    """Write all output DataFrames as Parquet and CSV to outputs/."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    pairs = [
        (trade_log, "trade_log"),
        (daily_signals, "daily_signals"),
        (portfolio_daily, "portfolio_daily"),
        (trade_summary, "trade_summary"),
    ]
    for df, name in pairs:
        df.to_parquet(OUTPUTS_DIR / f"{name}.parquet", index=False)
        df.to_csv(OUTPUTS_DIR / f"{name}.csv", index=False)
        logger.info("Wrote %s (%d rows)", name, len(df))


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest(features_d: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Execute the full backtest and return output DataFrames.

    Returns a dict with keys: trade_log, daily_signals, portfolio_daily,
    trade_summary. Also writes Parquet/CSV files to outputs/ and plots to
    outputs/plots/.
    """
    # ── Prepare input ────────────────────────────────────────────────────────
    df = features_d.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    _validate_input(df)

    # ── Initialise state ─────────────────────────────────────────────────────
    cash: float = INITIAL_CAPITAL
    positions: Dict[str, PositionState] = {}
    trade_logger = TradeLogger()

    daily_signals_rows: List[dict] = []
    portfolio_daily_rows: List[dict] = []
    trade_summary_rows: List[dict] = []

    # ── Main loop ─────────────────────────────────────────────────────────────
    for current_date, date_group in df.groupby("date", sort=True):
        today_closes: Dict[str, float] = (
            date_group.set_index("symbol")[EXECUTION_PRICE_COL].to_dict()
        )

        # BOD portfolio value — fixed for the entire day
        portfolio_value_bod = _portfolio_value(cash, positions, today_closes)

        for _, row in date_group.iterrows():
            ticker: str = str(row["symbol"])
            current_close: float = float(row[EXECUTION_PRICE_COL])

            raw_sl = row["sl_w"]
            stop_loss_price: float = float(raw_sl) if pd.notna(raw_sl) else math.nan

            buy_signal: int = int(row["buy"])

            raw_ema = row["ema21"]
            ema21: float = float(raw_ema) if pd.notna(raw_ema) else math.nan

            if ticker not in positions:
                positions[ticker] = PositionState()
            state = positions[ticker]

            # Capture BOD in-portfolio flag before any trade today
            in_portfolio_bod: bool = state.units_held > 0

            sell_signal: int = 0
            sell_executed: bool = False

            # ── SELL (stop-loss) ──────────────────────────────────────────────
            if (
                state.units_held > 0
                and not math.isnan(stop_loss_price)
                and current_close <= stop_loss_price
            ):
                sell_signal = 1
                units_sold = state.units_held
                proceeds = units_sold * current_close
                cost_basis = state.avg_buy_price * units_sold
                realized_gain_abs = proceeds - cost_basis
                realized_gain_pct = (
                    (proceeds / cost_basis - 1) if cost_basis > 0 else math.nan
                )
                sl_pct_sell = (
                    (current_close - stop_loss_price) / current_close
                )
                buy_date_for_ts = state.first_buy_date
                avg_bp_for_ts = state.avg_buy_price

                # Increment trade_count for this sell event before reset
                state.trade_count += 1
                trade_count_sell = state.trade_count

                cash += proceeds
                reset_position(state)

                trade_logger.log_event(
                    trade_count=trade_count_sell,
                    ticker=ticker,
                    event_type="SELL",
                    event_date=current_date.date(),
                    execution_price=current_close,
                    units_transacted=units_sold,
                    units_held_after=0,
                    avg_buy_price=math.nan,
                    stop_loss_price=stop_loss_price,
                    stop_loss_pct=sl_pct_sell,
                    position_size_used=proceeds,
                    portfolio_value=portfolio_value_bod,
                    cash_after=cash,
                    realized_gain_abs=realized_gain_abs,
                    realized_gain_pct=realized_gain_pct,
                )

                holding_days = (
                    (current_date.date() - buy_date_for_ts).days
                    if buy_date_for_ts is not None
                    else 0
                )
                trade_summary_rows.append(
                    {
                        "ticker": ticker,
                        "trade_count": trade_count_sell,
                        "buy_date": buy_date_for_ts,
                        "sell_date": current_date.date(),
                        "avg_buy_price": avg_bp_for_ts,
                        "sell_price": current_close,
                        "units": units_sold,
                        "cost_basis": cost_basis,
                        "proceeds": proceeds,
                        "realized_gain_abs": realized_gain_abs,
                        "realized_gain_pct": realized_gain_pct,
                        "holding_days": holding_days,
                    }
                )
                sell_executed = True

            # ── BUY ──────────────────────────────────────────────────────────
            if not sell_executed and buy_signal == 1:
                if math.isnan(stop_loss_price) or current_close <= 0:
                    logger.warning(
                        "%s %s: NaN/invalid stop_loss or close — skipping buy",
                        ticker, current_date,
                    )
                else:
                    # Determine if buy trigger fires (Case 1 or Case 2)
                    if state.units_held == 0:
                        should_buy = True  # Case 1: no existing position
                    else:
                        # Case 2: add-on — price must be rising
                        price_above_last = current_close > state.last_buy_price
                        price_above_ema = (
                            not math.isnan(ema21) and current_close > ema21
                        )
                        should_buy = price_above_last or price_above_ema

                    if should_buy:
                        # ── Allocation gate ────────────────────────────────
                        current_alloc = (
                            state.units_held * current_close / portfolio_value_bod
                            if portfolio_value_bod > 0 else 0.0
                        )
                        if current_alloc >= MAX_SECURITY_ALLOC:
                            logger.info(
                                "%s %s: max allocation reached (%.1f%% >= %.0f%%)",
                                ticker, current_date,
                                current_alloc * 100, MAX_SECURITY_ALLOC * 100,
                            )
                        elif cash < current_close:
                            logger.info(
                                "%s %s: insufficient cash (%.2f < %.2f)",
                                ticker, current_date, cash, current_close,
                            )
                        else:
                            try:
                                units_to_buy, raw_pos_size, sl_pct = compute_position_size(
                                    portfolio_value=portfolio_value_bod,
                                    current_close=current_close,
                                    stop_loss_price=stop_loss_price,
                                    units_held=state.units_held,
                                    ticker=ticker,
                                )
                            except ZeroDivisionError:
                                logger.warning(
                                    "%s %s: ZeroDivisionError in position sizing — skipping",
                                    ticker, current_date,
                                )
                                units_to_buy = 0
                                raw_pos_size = 0.0
                                sl_pct = 0.0

                            if units_to_buy > 0:
                                # Cap by available cash
                                cash_units = math.floor(cash / current_close)
                                units_to_buy = min(units_to_buy, cash_units)

                            if units_to_buy > 0:
                                cost = units_to_buy * current_close
                                cash -= cost
                                update_position_on_buy(
                                    state, units_to_buy, current_close, current_date.date()
                                )
                                trade_logger.log_event(
                                    trade_count=state.trade_count,
                                    ticker=ticker,
                                    event_type="BUY",
                                    event_date=current_date.date(),
                                    execution_price=current_close,
                                    units_transacted=units_to_buy,
                                    units_held_after=state.units_held,
                                    avg_buy_price=state.avg_buy_price,
                                    stop_loss_price=stop_loss_price,
                                    stop_loss_pct=sl_pct,
                                    position_size_used=cost,
                                    portfolio_value=portfolio_value_bod,
                                    cash_after=cash,
                                )

            # ── daily_signals row ─────────────────────────────────────────────
            if not math.isnan(stop_loss_price) and current_close > stop_loss_price and current_close > 0:
                sl_pct_ds = (current_close - stop_loss_price) / current_close
                pos_size_est = (MAX_RISK_PER_TRADE * portfolio_value_bod) / sl_pct_ds
                units_purch_est = math.floor(pos_size_est / current_close)
            else:
                sl_pct_ds = (
                    (current_close - stop_loss_price) / current_close
                    if not math.isnan(stop_loss_price) and current_close > 0
                    else math.nan
                )
                pos_size_est = math.nan
                units_purch_est = 0

            daily_signals_rows.append(
                {
                    "date": current_date.date(),
                    "ticker": ticker,
                    "close": current_close,
                    "sl_w": stop_loss_price if not math.isnan(stop_loss_price) else None,
                    "stop_loss_pct": sl_pct_ds,
                    "buy_signal": buy_signal,
                    "sell_signal": sell_signal,
                    "in_portfolio": in_portfolio_bod,
                    "units_held": state.units_held,
                    "avg_buy_price": state.avg_buy_price if state.units_held > 0 else math.nan,
                    "position_size_estimate": pos_size_est,
                    "units_purchaseable": units_purch_est,
                    "portfolio_value_bod": portfolio_value_bod,
                }
            )

        # ── End-of-day portfolio snapshot ─────────────────────────────────────
        pv_eod = _portfolio_value(cash, positions, today_closes)
        invested_eod = pv_eod - cash
        portfolio_daily_rows.append(
            {
                "date": current_date.date(),
                "portfolio_value": pv_eod,
                "cash": cash,
                "invested_value": invested_eod,
            }
        )

    # ── Post-processing ───────────────────────────────────────────────────────
    trade_log = trade_logger.to_dataframe()

    daily_signals = pd.DataFrame(daily_signals_rows)

    portfolio_daily = pd.DataFrame(portfolio_daily_rows)
    if len(portfolio_daily) > 0:
        pv = portfolio_daily["portfolio_value"]
        portfolio_daily["daily_return"] = pv.pct_change()
        portfolio_daily["cumulative_return"] = pv / pv.iloc[0] - 1

    trade_summary = pd.DataFrame(trade_summary_rows) if trade_summary_rows else pd.DataFrame(
        columns=[
            "ticker", "trade_count", "buy_date", "sell_date",
            "avg_buy_price", "sell_price", "units", "cost_basis",
            "proceeds", "realized_gain_abs", "realized_gain_pct", "holding_days",
        ]
    )

    # ── Write outputs ─────────────────────────────────────────────────────────
    _write_outputs(trade_log, daily_signals, portfolio_daily, trade_summary)

    # ── Analytics ─────────────────────────────────────────────────────────────
    summary_stats = analytics.compute_summary_stats(portfolio_daily, trade_summary)
    logger.info("Backtest complete. Summary: %s", summary_stats)

    analytics.plot_portfolio_curve(portfolio_daily)
    analytics.plot_drawdown(portfolio_daily)
    analytics.plot_trade_distribution(trade_summary)
    analytics.plot_by_ticker(trade_summary)

    return {
        "trade_log": trade_log,
        "daily_signals": daily_signals,
        "portfolio_daily": portfolio_daily,
        "trade_summary": trade_summary,
        "summary_stats": summary_stats,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    data_path = Path("data/features/features_d.csv")
    if not data_path.exists():
        logger.error("features_d.csv not found at %s", data_path)
        sys.exit(1)

    logger.info("Loading features_d from %s", data_path)
    features_d = pd.read_csv(data_path)
    results = run_backtest(features_d)

    pd.set_option("display.float_format", "{:,.2f}".format)
    stats = results["summary_stats"]
    print("\n=== Backtest Summary ===")
    for k, v in stats.items():
        print(f"  {k:<40} {v}")
