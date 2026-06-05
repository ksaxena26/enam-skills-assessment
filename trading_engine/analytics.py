import logging
from pathlib import Path
from typing import Dict

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
