import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PLOTS_DIR = Path("outputs") / "plots"


def compute_summary_stats(
    portfolio_daily: pd.DataFrame,
    trade_summary: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute portfolio performance statistics.

    Returns a dict with keys: total_trades, win_rate, avg_gain_pct,
    median_gain_pct, total_realized_gain_abs, portfolio_cumulative_return_pct,
    annualized_return_pct, sharpe_ratio, max_drawdown_pct, calmar_ratio,
    avg_holding_days.
    """
    stats: Dict[str, float] = {}

    stats["total_trades"] = int(len(trade_summary))

    if len(trade_summary) > 0:
        stats["win_rate"] = float((trade_summary["realized_gain_pct"] > 0).mean())
        stats["avg_gain_pct"] = float(trade_summary["realized_gain_pct"].mean())
        stats["median_gain_pct"] = float(trade_summary["realized_gain_pct"].median())
        stats["total_realized_gain_abs"] = float(trade_summary["realized_gain_abs"].sum())
        stats["avg_holding_days"] = float(trade_summary["holding_days"].mean())
    else:
        stats.update(
            win_rate=float("nan"),
            avg_gain_pct=float("nan"),
            median_gain_pct=float("nan"),
            total_realized_gain_abs=0.0,
            avg_holding_days=float("nan"),
        )

    if len(portfolio_daily) >= 2:
        pv = portfolio_daily["portfolio_value"].astype(float)
        start_val = pv.iloc[0]
        end_val = pv.iloc[-1]

        stats["portfolio_cumulative_return_pct"] = float((end_val / start_val - 1) * 100)

        start_date = pd.to_datetime(portfolio_daily["date"].iloc[0])
        end_date = pd.to_datetime(portfolio_daily["date"].iloc[-1])
        calendar_days = max((end_date - start_date).days, 1)
        stats["annualized_return_pct"] = float(
            ((end_val / start_val) ** (365.0 / calendar_days) - 1) * 100
        )

        daily_ret = portfolio_daily["daily_return"].dropna().astype(float)
        if len(daily_ret) > 1 and daily_ret.std() > 0:
            stats["sharpe_ratio"] = float(
                daily_ret.mean() / daily_ret.std() * np.sqrt(252)
            )
        else:
            stats["sharpe_ratio"] = float("nan")

        roll_max = pv.cummax()
        drawdown = (pv - roll_max) / roll_max
        stats["max_drawdown_pct"] = float(drawdown.min() * 100)

        ann_ret = stats["annualized_return_pct"]
        mdd = stats["max_drawdown_pct"]
        stats["calmar_ratio"] = float(ann_ret / abs(mdd)) if mdd < 0 else float("nan")
    else:
        stats.update(
            portfolio_cumulative_return_pct=float("nan"),
            annualized_return_pct=float("nan"),
            sharpe_ratio=float("nan"),
            max_drawdown_pct=float("nan"),
            calmar_ratio=float("nan"),
        )

    return stats


def plot_portfolio_curve(portfolio_daily: pd.DataFrame) -> plt.Figure:
    """Cumulative return curve over time. Saves PNG to outputs/plots/."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        pd.to_datetime(portfolio_daily["date"]),
        portfolio_daily["cumulative_return"].astype(float) * 100,
        linewidth=1.5,
        color="steelblue",
    )
    ax.set_title("Portfolio Cumulative Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "portfolio_curve.png", dpi=150)
    logger.info("Saved portfolio_curve.png")
    return fig


def plot_drawdown(portfolio_daily: pd.DataFrame) -> plt.Figure:
    """Rolling drawdown from peak over time. Saves PNG to outputs/plots/."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    pv = portfolio_daily["portfolio_value"].astype(float)
    roll_max = pv.cummax()
    drawdown = (pv - roll_max) / roll_max * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(
        pd.to_datetime(portfolio_daily["date"]),
        drawdown.values,
        0,
        alpha=0.6,
        color="crimson",
    )
    ax.set_title("Portfolio Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "drawdown.png", dpi=150)
    logger.info("Saved drawdown.png")
    return fig


def plot_trade_distribution(trade_summary: pd.DataFrame) -> plt.Figure:
    """Histogram of realized_gain_pct across all closed trades. Saves PNG."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(trade_summary) > 0:
        ax.hist(
            trade_summary["realized_gain_pct"].astype(float) * 100,
            bins=30,
            edgecolor="black",
            alpha=0.8,
            color="steelblue",
        )
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax.set_title("Trade Return Distribution")
    ax.set_xlabel("Realized Gain (%)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "trade_distribution.png", dpi=150)
    logger.info("Saved trade_distribution.png")
    return fig


def plot_by_ticker(trade_summary: pd.DataFrame) -> plt.Figure:
    """Bar chart of total realized gain per ticker, sorted descending. Saves PNG."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    if len(trade_summary) > 0:
        by_ticker = (
            trade_summary.groupby("ticker")["realized_gain_abs"]
            .sum()
            .sort_values(ascending=False)
        )
        colors = ["green" if v >= 0 else "crimson" for v in by_ticker.values]
        ax.bar(range(len(by_ticker)), by_ticker.values, color=colors, alpha=0.8)
        ax.set_xticks(range(len(by_ticker)))
        ax.set_xticklabels(by_ticker.index, rotation=90, fontsize=7)
    ax.set_title("Total Realized Gain by Ticker")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Realized Gain (abs)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "gain_by_ticker.png", dpi=150)
    logger.info("Saved gain_by_ticker.png")
    return fig


def plot_trades_candlestick(
    trade_log: pd.DataFrame,
    features_d: pd.DataFrame,
    daily_signals: Optional[pd.DataFrame] = None,
) -> List[Any]:
    """
    One interactive Plotly candlestick chart per traded ticker.

    Shows OHLC candles, EMA21, a dotted stop-loss line for each holding period,
    and buy/sell markers. Saves .html (always) and .png (requires kaleido) to
    outputs/plots/candlestick_{ticker}.{ext}.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("plotly not installed — run: pip install plotly")
        return []

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if len(trade_log) == 0:
        logger.info("No trades in trade_log — skipping candlestick charts")
        return []

    has_ohlc = all(c in features_d.columns for c in ["open", "high", "low"])
    has_volume = "volume" in features_d.columns
    figures: List[Any] = []

    for ticker in sorted(trade_log["ticker"].unique()):
        # ── OHLC ──────────────────────────────────────────────────────────────
        ohlc = (
            features_d[features_d["symbol"] == ticker]
            .copy()
            .assign(date=lambda d: pd.to_datetime(d["date"]))
            .sort_values("date")
            .reset_index(drop=True)
        )
        if len(ohlc) == 0:
            logger.warning("No price data for %s — skipping", ticker)
            continue

        # ── Trade events ───────────────────────────────────────────────────────
        ticker_tl = trade_log[trade_log["ticker"] == ticker].copy()
        ticker_tl["date"] = pd.to_datetime(ticker_tl["date"])
        buys = ticker_tl[ticker_tl["event_type"] == "BUY"]
        sells = ticker_tl[ticker_tl["event_type"] == "SELL"]

        # ── Subplots (add volume row if available) ────────────────────────────
        rows = 2 if has_volume else 1
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25] if has_volume else [1.0],
            vertical_spacing=0.03,
        )

        # ── Candles / close line ───────────────────────────────────────────────
        if has_ohlc:
            fig.add_trace(
                go.Candlestick(
                    x=ohlc["date"],
                    open=ohlc["open"], high=ohlc["high"],
                    low=ohlc["low"],   close=ohlc["close"],
                    name="Price",
                    increasing_line_color="#26a69a", increasing_fillcolor="#26a69a",
                    decreasing_line_color="#ef5350", decreasing_fillcolor="#ef5350",
                ),
                row=1, col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=ohlc["date"], y=ohlc["close"],
                    name="Close", line=dict(color="#26a69a", width=1),
                ),
                row=1, col=1,
            )

        # ── EMA21 ──────────────────────────────────────────────────────────────
        if "ema21" in ohlc.columns:
            fig.add_trace(
                go.Scatter(
                    x=ohlc["date"], y=ohlc["ema21"],
                    name="EMA 21",
                    line=dict(color="#ffb300", width=1.2),
                    opacity=0.9,
                ),
                row=1, col=1,
            )

        # ── Stop-loss — separate trace per contiguous holding period ──────────
        if daily_signals is not None and len(daily_signals) > 0:
            ds_t = (
                daily_signals[daily_signals["ticker"] == ticker]
                .copy()
                .assign(date=lambda d: pd.to_datetime(d["date"]))
                .sort_values("date")
                .reset_index(drop=True)
            )
            if len(ds_t) > 0:
                # Each uninterrupted in_portfolio=True block gets a unique id
                ds_t["_pid"] = (~ds_t["in_portfolio"].astype(bool)).cumsum()
                held = ds_t[ds_t["in_portfolio"].astype(bool)]
                sl_in_legend = False
                for _pid, grp in held.groupby("_pid", sort=True):
                    valid = grp.dropna(subset=["sl_w"])
                    if len(valid) == 0:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=valid["date"],
                            y=valid["sl_w"].astype(float),
                            mode="lines",
                            name="Stop Loss",
                            line=dict(color="#ff5252", width=1.5, dash="dot"),
                            legendgroup="sl",
                            showlegend=not sl_in_legend,
                            opacity=0.8,
                        ),
                        row=1, col=1,
                    )
                    sl_in_legend = True

        # ── Buy markers ────────────────────────────────────────────────────────
        if len(buys) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buys["date"], y=buys["execution_price"],
                    mode="markers+text",
                    name="Buy",
                    text=["B"] * len(buys),
                    textposition="bottom center",
                    textfont=dict(size=8, color="white"),
                    marker=dict(
                        symbol="triangle-up", size=14,
                        color="#1565c0",
                        line=dict(width=1, color="white"),
                    ),
                ),
                row=1, col=1,
            )

        # ── Sell markers ───────────────────────────────────────────────────────
        if len(sells) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sells["date"], y=sells["execution_price"],
                    mode="markers+text",
                    name="Sell",
                    text=["S"] * len(sells),
                    textposition="top center",
                    textfont=dict(size=8, color="white"),
                    marker=dict(
                        symbol="triangle-down", size=14,
                        color="#b71c1c",
                        line=dict(width=1, color="white"),
                    ),
                ),
                row=1, col=1,
            )

        # ── Volume bars ────────────────────────────────────────────────────────
        if has_volume:
            bar_colors = [
                "#26a69a" if c >= o else "#ef5350"
                for c, o in zip(
                    ohlc["close"],
                    ohlc["open"] if has_ohlc else ohlc["close"],
                )
            ]
            fig.add_trace(
                go.Bar(
                    x=ohlc["date"], y=ohlc["volume"],
                    name="Volume", marker_color=bar_colors,
                    opacity=0.5, showlegend=False,
                ),
                row=2, col=1,
            )

        # ── Title with trade summary ───────────────────────────────────────────
        n_b, n_s = len(buys), len(sells)
        gain_str = ""
        if n_s > 0 and "realized_gain_pct" in sells.columns:
            net = sells["realized_gain_pct"].sum() * 100
            gain_str = f"  |  Realized: {net:+.1f}%"

        fig.update_layout(
            title=dict(
                text=f"<b>{ticker}</b>  —  {n_b} buy{'s' if n_b != 1 else ''}, "
                     f"{n_s} sell{'s' if n_s != 1 else ''}{gain_str}",
                font=dict(size=15),
            ),
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=600 if not has_volume else 720,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            ),
            margin=dict(t=80, b=40, l=60, r=40),
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        if has_volume:
            fig.update_yaxes(title_text="Volume", showticklabels=False, row=2, col=1)

        # ── Save HTML (always) ────────────────────────────────────────────────
        html_path = PLOTS_DIR / f"candlestick_{ticker}.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        logger.info("Saved candlestick_%s.html", ticker)

        # ── Save PNG (needs kaleido) ───────────────────────────────────────────
        try:
            png_path = PLOTS_DIR / f"candlestick_{ticker}.png"
            fig.write_image(
                str(png_path),
                width=1400,
                height=600 if not has_volume else 720,
                scale=1.5,
            )
            logger.info("Saved candlestick_%s.png", ticker)
        except Exception as exc:
            logger.warning(
                "PNG skipped for %s (pip install kaleido for static export): %s",
                ticker, exc,
            )

        figures.append(fig)

    return figures
