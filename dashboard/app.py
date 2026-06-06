"""
Trading Engine Dashboard — Streamlit web app.

Local:   streamlit run dashboard/app.py
EC2:     streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0

Environment variables
---------------------
  DASHBOARD_PASSWORD   plain-text login password   (default: demo123)
"""
import math
import os
import sys
from pathlib import Path
from typing import Optional

# ── Project root ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from trading_engine import analytics
from trading_engine.config import INITIAL_CAPITAL, MAX_RISK_PER_TRADE, MAX_SECURITY_ALLOC
from trading_engine.engine import run_backtest

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wisdom Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auth ───────────────────────────────────────────────────────────────────────
_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "demo123")
_LOGO_PATH = Path(__file__).parent / "logo.svg"


def _logo_html(width: int = 160) -> str:
    """Inline base64 <img> for the SVG logo — renders in sidebar and main area."""
    if _LOGO_PATH.exists():
        import base64
        b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
        return (
            f'<div style="text-align:center;margin-bottom:6px">'
            f'<img src="data:image/svg+xml;base64,{b64}" width="{width}"/>'
            f'</div>'
        )
    return ""


def _require_auth() -> bool:
    if st.session_state.get("authenticated"):
        return True
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown(_logo_html(180), unsafe_allow_html=True)
        st.markdown("## Wisdom Trader")
        st.markdown("---")
        # Wrapping in st.form makes the Enter key submit the password
        with st.form("login_form"):
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Login", use_container_width=True):
                if pw == _PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
    return False


# ── Defaults ───────────────────────────────────────────────────────────────────
_DEFAULT_PARAMS = dict(
    max_security_alloc=MAX_SECURITY_ALLOC,
    max_risk_per_trade=MAX_RISK_PER_TRADE,
    atr_mult=1.00,
    st_buffer=0.020,
    overshoot_coeff=1.15,
    crsi_threshold=50.0,
    risk_threshold=0.40,
)

_OPTUNA_BEST = dict(
    max_security_alloc=0.899752,
    max_risk_per_trade=0.037807,
    atr_mult=2.374657,
    st_buffer=0.118153,
    overshoot_coeff=1.220966,
    crsi_threshold=44.49437,
    risk_threshold=0.286903,
)

for _k, _v in dict(authenticated=False, features_d=None, backtest_results=None,
                   backtest_error=None, params=_DEFAULT_PARAMS.copy()).items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Formatting ─────────────────────────────────────────────────────────────────
def _inr(val: float) -> str:
    if not math.isfinite(val):
        return "—"
    if abs(val) >= 1e7:
        return f"₹{val / 1e7:.2f} Cr"
    if abs(val) >= 1e5:
        return f"₹{val / 1e5:.1f} L"
    return f"₹{val:,.0f}"


def _pct(val: float, decimals: int = 2) -> str:
    return f"{val:.{decimals}f}%" if math.isfinite(val) else "—"


# ── Raw-data availability ──────────────────────────────────────────────────────
def _raw_data_available() -> bool:
    """True when raw OHLCV CSVs are present for the full engineering pipeline.
    Checks the filesystem directly — avoids importing data_manager which calls
    Azure Key Vault at module level and fails outside the user's credential context."""
    ohlcv_dir = _ROOT / "data" / "ohlcv"
    return ohlcv_dir.exists() and any(ohlcv_dir.glob("*.csv"))


# ── Signal tables ──────────────────────────────────────────────────────────────
def _build_signal_tables(
    fd: pd.DataFrame,
    date_str: str,
    portfolio_value: float,
    daily_signals: Optional[pd.DataFrame],
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (buy_df, sell_df) for the given YYYY-MM-DD date.

    Uses daily_signals.portfolio_value_bod for per-ticker sizing when available,
    falls back to the supplied portfolio_value otherwise.
    """
    fd_copy = fd.copy()
    fd_copy["_ds"] = pd.to_datetime(fd_copy["date"]).dt.strftime("%Y-%m-%d")
    day = fd_copy[fd_copy["_ds"] == date_str]

    # per-ticker portfolio value from backtest (more accurate sizing)
    pv_map: dict = {}
    in_portfolio: set = set()
    if daily_signals is not None and len(daily_signals) > 0:
        ds = daily_signals.copy()
        ds["_ds"] = pd.to_datetime(ds["date"]).dt.strftime("%Y-%m-%d")
        ds_day = ds[ds["_ds"] == date_str]
        if len(ds_day) > 0:
            pv_map = ds_day.set_index("ticker")["portfolio_value_bod"].to_dict()
            in_portfolio = set(
                ds_day.loc[ds_day["in_portfolio"].astype(bool), "ticker"].tolist()
            )

    mrt = params["max_risk_per_trade"]

    buy_rows = []
    for _, row in day[day["buy"] == 1].iterrows():
        close = float(row["close"])
        sl = float(row["sl_w"]) if pd.notna(row["sl_w"]) else None
        if sl is None or sl <= 0 or close <= sl:
            continue
        sl_pct = (close - sl) / close
        pv = pv_map.get(row["symbol"], portfolio_value)
        pos_size = (mrt * pv) / sl_pct if sl_pct > 0 else 0.0
        units = math.floor(pos_size / close) if close > 0 else 0
        buy_rows.append({
            "Ticker": row["symbol"],
            "CMP (₹)": f"{close:,.2f}",
            "Stop Loss (₹)": f"{sl:,.2f}",
            "Max Risk %": f"{sl_pct * 100:.2f}",
            "Position Size": _inr(pos_size),
            "Units": units,
        })

    # Sell alerts: price at/below stop-loss for in-portfolio securities.
    # If no backtest result, flag ALL stop-loss breaches as potential alerts.
    show_all = len(in_portfolio) == 0 and daily_signals is None
    sell_rows = []
    for _, row in day.iterrows():
        close = float(row["close"])
        sl = float(row["sl_w"]) if pd.notna(row["sl_w"]) else None
        sym = row["symbol"]
        if sl is not None and close <= sl and (show_all or sym in in_portfolio):
            loss_pct = abs((close - sl) / sl * 100) if sl > 0 else 0
            sell_rows.append({
                "Ticker": sym,
                "CMP (₹)": f"{close:,.2f}",
                "Stop Loss (₹)": f"{sl:,.2f}",
                "Breach %": f"{loss_pct:.1f}",
            })

    return pd.DataFrame(buy_rows), pd.DataFrame(sell_rows)


# ── Plotly helpers ─────────────────────────────────────────────────────────────
def _portfolio_fig(portfolio_daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=pd.to_datetime(portfolio_daily["date"]),
        y=portfolio_daily["cumulative_return"].astype(float) * 100,
        mode="lines", line=dict(color="#26a69a", width=1.5), name="Cumulative Return %",
    ))
    fig.update_layout(
        title="Cumulative Return", xaxis_title="Date", yaxis_title="Return (%)",
        template="plotly_dark", height=340,
        margin=dict(t=40, b=30, l=55, r=20), hovermode="x unified",
    )
    return fig


def _drawdown_fig(portfolio_daily: pd.DataFrame) -> go.Figure:
    pv = portfolio_daily["portfolio_value"].astype(float)
    dd = (pv - pv.cummax()) / pv.cummax() * 100
    fig = go.Figure(go.Scatter(
        x=pd.to_datetime(portfolio_daily["date"]), y=dd,
        fill="tozeroy", fillcolor="rgba(239,83,80,0.25)",
        line=dict(color="#ef5350", width=1), name="Drawdown %",
    ))
    fig.update_layout(
        title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown (%)",
        template="plotly_dark", height=260,
        margin=dict(t=40, b=30, l=55, r=20), hovermode="x unified",
    )
    return fig


# ── Pipeline helpers ───────────────────────────────────────────────────────────
def _patch_engine(params: dict) -> None:
    import trading_engine.engine as _eng
    import trading_engine.position_manager as _pm
    _eng.MAX_SECURITY_ALLOC = params["max_security_alloc"]
    _eng.MAX_RISK_PER_TRADE = params["max_risk_per_trade"]
    _pm.MAX_SECURITY_ALLOC = params["max_security_alloc"]
    _pm.MAX_RISK_PER_TRADE = params["max_risk_per_trade"]


def _restore_engine() -> None:
    import trading_engine.engine as _eng
    import trading_engine.position_manager as _pm
    import trading_engine.config as _cfg
    _eng.MAX_SECURITY_ALLOC = _cfg.MAX_SECURITY_ALLOC
    _eng.MAX_RISK_PER_TRADE = _cfg.MAX_RISK_PER_TRADE
    _pm.MAX_SECURITY_ALLOC = _cfg.MAX_SECURITY_ALLOC
    _pm.MAX_RISK_PER_TRADE = _cfg.MAX_RISK_PER_TRADE


def _run_backtest(fd: pd.DataFrame, params: dict) -> dict:
    """Run backtest with patched params, restore afterward. Returns results dict."""
    import traceback
    _patch_engine(params)
    try:
        results = run_backtest(fd, write_outputs=True, run_analytics=False)
    except Exception as exc:
        return {"error": traceback.format_exc()}
    finally:
        _restore_engine()

    results["features_d"] = fd

    # Candlestick generation is best-effort — don't let it block result display
    if len(results["trade_log"]) > 0:
        try:
            results["candlestick_figs"] = analytics.plot_trades_candlestick(
                results["trade_log"],
                fd,
                results.get("daily_signals"),
            )
            results["candlestick_tickers"] = sorted(
                results["trade_log"]["ticker"].unique()
            )
        except Exception as exc:
            results["candlestick_figs"] = []
            results["candlestick_tickers"] = []
            results["candlestick_error"] = str(exc)
    else:
        results["candlestick_figs"] = []
        results["candlestick_tickers"] = []

    return results


def _load_ohlcv() -> pd.DataFrame:
    """Read OHLCV CSVs directly — avoids data_manager's msoffcrypto/Azure imports."""
    ohlcv_dir = _ROOT / "data" / "ohlcv"
    frames = [pd.read_csv(f) for f in sorted(ohlcv_dir.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.drop_duplicates(["date", "symbol"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def _run_full_pipeline(params: dict) -> dict:
    """Full engineering pipeline + backtest. Returns results dict or error dict."""
    try:
        from src.engineering import (
            compute_buy_range_u, compute_features_d, compute_features_m,
            compute_features_w, compute_max_risk, get_batches,
            map_monthly_swings_to_daily, map_weekly_to_daily,
        )
        from src.utils.utils import daily_to_monthly_transform, daily_to_weekly_transform
    except Exception as exc:
        return {"error": f"Import failed: {exc}"}

    try:
        price_d = _load_ohlcv()
        price_w = daily_to_weekly_transform(price_d).drop(columns=["prev_close"])
        price_m = daily_to_monthly_transform(price_d).drop(columns=["prev_close"])
        price_d = price_d.drop(columns=["prev_close"])

        fd = pd.concat([compute_features_d(b) for b in get_batches(price_d)], ignore_index=True)

        fw_list = []
        for b in get_batches(price_w):
            fw = compute_features_w(b)
            for sym, idx in fw.groupby("symbol").groups.items():
                grp = fw.loc[idx]
                sl1 = grp[["sma30", "wdma30"]].min(axis=1) - params["atr_mult"] * grp["atr_14"]
                sl2 = (1 - params["st_buffer"]) * grp["supert"]
                sl3 = (1 - params["st_buffer"]) * grp["supert_b1"]
                sl = pd.concat([sl1, sl2, sl3], axis=1).min(axis=1)
                fw.loc[idx, "sl"] = sl.values
                fw.loc[idx, "sl_b1"] = sl.shift(1).values
            fw_list.append(fw)
        fw = pd.concat(fw_list, ignore_index=True)

        fm_list = []
        for b in get_batches(price_m):
            fm = compute_features_m(b)
            for sym, idx in fm.groupby("symbol").groups.items():
                grp = fm.loc[idx]
                mbp = params["overshoot_coeff"] * grp["peak"]
                fm.loc[idx, "max_buy_price"] = mbp.values
                fm.loc[idx, "max_buy_price_b1"] = mbp.shift(1).values
            fm_list.append(fm)
        fm = pd.concat(fm_list, ignore_index=True)

        fd = map_weekly_to_daily(fd, fw, ["supert", "sl"])
        fd = map_monthly_swings_to_daily(fd, fm)
        fd = compute_buy_range_u(fd, fm)
        fd["risk"] = compute_max_risk(fd, "sl_w")

        conds = [
            fd["close"] > fd["supert_w"],
            fd["crsi"] < params["crsi_threshold"],
            fd["risk"] < params["risk_threshold"],
            fd["close"] < fd["max_buy_price_b1"],
        ]
        fd["buy"] = np.where(np.all(conds, axis=0), 1, 0)
        fd = fd[fd["symbol"] != "NIFTYBEES"].copy()

    except Exception as exc:
        return {"error": f"Engineering pipeline failed: {exc}"}

    return _run_backtest(fd, params)


# ── Main app ───────────────────────────────────────────────────────────────────
def main() -> None:
    if not _require_auth():
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(_logo_html(140), unsafe_allow_html=True)
        st.title("Wisdom Trader")
        st.markdown("---")

        # Auto-load from disk on first visit if no data is in session yet
        _DEFAULT_DATA = Path("data/features/features_d.csv")
        if st.session_state.features_d is None and _DEFAULT_DATA.exists():
            try:
                _fd = pd.read_csv(_DEFAULT_DATA)
                _fd["date"] = pd.to_datetime(_fd["date"])
                st.session_state.features_d = _fd
                st.session_state["_uploaded_file_id"] = "__preloaded__"
            except Exception:
                pass  # fall through to manual upload

        uploaded = st.file_uploader(
            "Upload features_d.csv",
            type=["csv"],
            help="Replaces the pre-loaded data. Use to refresh with a newer file.",
        )
        if uploaded is not None:
            # Only re-parse when a different file is selected (file_id is stable
            # across reruns for the same upload — avoids resetting results every rerun)
            if st.session_state.get("_uploaded_file_id") != uploaded.file_id:
                try:
                    fd = pd.read_csv(uploaded)
                    fd["date"] = pd.to_datetime(fd["date"])
                    st.session_state.features_d = fd
                    st.session_state.backtest_results = None
                    st.session_state.backtest_error = None
                    st.session_state["_uploaded_file_id"] = uploaded.file_id
                    st.success(f"Loaded {len(fd):,} rows · {fd['symbol'].nunique()} tickers")
                except Exception as exc:
                    st.error(f"Failed to parse CSV: {exc}")

        if st.session_state.features_d is not None:
            fd_sidebar = st.session_state.features_d
            min_d = fd_sidebar["date"].min().date()
            max_d = fd_sidebar["date"].max().date()
            n_buy = int((fd_sidebar["buy"] == 1).sum())
            st.caption(f"**Range:** {min_d} → {max_d}")
            st.caption(f"**Buy signals:** {n_buy:,}")

        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    if st.session_state.features_d is None:
        st.info("No data loaded. Upload `features_d.csv` from the sidebar to get started.")
        return

    fd: pd.DataFrame = st.session_state.features_d
    results: Optional[dict] = st.session_state.backtest_results
    params: dict = st.session_state.params

    tab1, tab2, tab3 = st.tabs(["Daily Signals", "Backtest", "Parameters"])

    # ── TAB 1: Daily Signals ──────────────────────────────────────────────────
    with tab1:
        import datetime as _dt
        _available_dates = sorted(fd["date"].dt.date.unique())
        _min_d, _max_d = _available_dates[0], _available_dates[-1]
        _default_d = _available_dates[-1]  # most recent date
        picked = st.date_input(
            "Select date",
            value=_default_d,
            min_value=_min_d,
            max_value=_max_d,
            key="date_pick",
        )
        # Clamp to nearest available trading date if user picks a non-trading day
        if picked not in _available_dates:
            picked = min(_available_dates, key=lambda d: abs((d - picked).days))
            st.caption(f"No data for that date — showing nearest trading day: {picked}")
        sel_date = picked.strftime("%Y-%m-%d")

        # Portfolio value for sizing: use backtest value if available
        pv_for_sizing = float(INITIAL_CAPITAL)
        if results:
            pd_df = results["portfolio_daily"]
            pd_df_d = pd_df[pd_df["date"].astype(str).str[:10] == sel_date]
            if len(pd_df_d) > 0:
                pv_for_sizing = float(pd_df_d.iloc[0]["portfolio_value"])

        daily_signals = results["daily_signals"] if results else None
        buy_df, sell_df = _build_signal_tables(fd, sel_date, pv_for_sizing, daily_signals, params)

        col_b, col_s = st.columns([3, 2])

        with col_b:
            st.markdown(f"#### Buy Signals — {sel_date}")
            if not results:
                st.caption(
                    f"Sizing uses initial capital {_inr(INITIAL_CAPITAL)}. "
                    "Run backtest for portfolio-aware values."
                )
            if len(buy_df) > 0:
                st.dataframe(buy_df, use_container_width=True, hide_index=True)
            else:
                st.info("No buy signals for this date.")

        with col_s:
            st.markdown(f"#### Sell Alerts — {sel_date}")
            if not results:
                st.caption("Showing all stop-loss breaches (no position filter — backtest not run).")
            if len(sell_df) > 0:
                st.dataframe(sell_df, use_container_width=True, hide_index=True)
            else:
                st.info("No stop-loss alerts.")

    # ── TAB 2: Backtest Results ────────────────────────────────────────────────
    with tab2:
        if results is None:
            st.info("No backtest results yet. Click below to run with current parameters, "
                    "or adjust parameters in the Parameters tab first.")
            if st.button("Run Backtest", type="primary", use_container_width=True):
                st.session_state.backtest_error = None
                with st.spinner("Running backtest…"):
                    res = _run_backtest(fd, params)
                if "error" in res:
                    st.session_state.backtest_error = res["error"]
                else:
                    st.session_state.backtest_results = res
                    st.rerun()

            # Show error persistently (survives reruns via session_state)
            if st.session_state.get("backtest_error"):
                st.error("Backtest failed:")
                st.code(st.session_state.backtest_error)
        else:
            stats = results["summary_stats"]

            # ── Summary metrics ────────────────────────────────────────────────
            def _safe_pct(key: str, mult: float = 1.0, dec: int = 2) -> str:
                v = stats.get(key, float("nan"))
                return _pct(float(v) * mult, dec) if math.isfinite(float(v)) else "—"

            def _safe_f(key: str, fmt: str = ".3f") -> str:
                v = stats.get(key, float("nan"))
                return format(float(v), fmt) if math.isfinite(float(v)) else "—"

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("CAGR", _safe_pct("annualized_return_pct"))
            c2.metric("Cumulative Return", _safe_pct("portfolio_cumulative_return_pct", dec=1))
            c3.metric("Sharpe Ratio", _safe_f("sharpe_ratio"))
            c4.metric("Max Drawdown", _safe_pct("max_drawdown_pct", dec=1))
            c5.metric("Win Rate", _safe_pct("win_rate", mult=100, dec=1))

            st.markdown("---")

            # ── Charts + trade summary ─────────────────────────────────────────
            col_l, col_r = st.columns([3, 2])

            with col_l:
                st.plotly_chart(_portfolio_fig(results["portfolio_daily"]), use_container_width=True)
                st.plotly_chart(_drawdown_fig(results["portfolio_daily"]), use_container_width=True)

            with col_r:
                st.markdown("#### Closed Trades")
                ts = results["trade_summary"]
                if len(ts) > 0:
                    disp = ts[["ticker", "buy_date", "sell_date", "avg_buy_price",
                                "sell_price", "realized_gain_pct", "holding_days"]].copy()
                    disp["realized_gain_pct"] = (
                        disp["realized_gain_pct"].astype(float) * 100
                    ).map(lambda x: f"{x:+.1f}%")
                    disp.columns = ["Ticker", "Buy Date", "Sell Date", "Avg Buy", "Sell", "Gain", "Days"]
                    st.dataframe(disp, use_container_width=True, hide_index=True)
                else:
                    st.info("No closed trades.")

                st.markdown("#### Summary")
                stat_rows = [
                    ("Total trades",     str(int(stats.get("total_trades", 0)))),
                    ("Win rate",         _safe_pct("win_rate", mult=100, dec=1)),
                    ("Avg gain",         _safe_pct("avg_gain_pct", mult=100, dec=1)),
                    ("Median gain",      _safe_pct("median_gain_pct", mult=100, dec=1)),
                    ("Avg hold",         f"{_safe_f('avg_holding_days', '.0f')} days"),
                    ("Calmar",           _safe_f("calmar_ratio")),
                    ("Total P&L",        _inr(float(stats.get("total_realized_gain_abs", 0.0)))),
                ]
                for label, val in stat_rows:
                    st.markdown(f"**{label}:** {val}")

            # ── Candlestick charts ─────────────────────────────────────────────
            figs = results.get("candlestick_figs", [])
            tickers = results.get("candlestick_tickers", [])
            if figs and tickers:
                st.markdown("---")
                st.markdown("#### Trade Charts")
                sel_ticker = st.selectbox("Ticker", tickers, key="candle_sel")
                if sel_ticker in tickers:
                    idx = tickers.index(sel_ticker)
                    st.plotly_chart(figs[idx], use_container_width=True)

    # ── TAB 3: Parameters ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Strategy Parameters")

        raw_ok = _raw_data_available()
        if not raw_ok:
            st.info(
                "Raw OHLC data not found. Engine parameters (max_security_alloc, "
                "max_risk_per_trade) can be applied directly. Engineering parameters "
                "(atr_mult, st_buffer, overshoot_coeff, crsi_threshold, risk_threshold) "
                "require the full data pipeline — not available in this environment."
            )

        hint_col, _ = st.columns([2, 3])
        with hint_col:
            if st.button("Load Optuna Best  (Trial #348 · CAGR 16.72%)"):
                st.session_state.params = _OPTUNA_BEST.copy()
                st.rerun()

        st.markdown("---")

        with st.form("params_form"):
            st.markdown("**Engine — position sizing and allocation**")
            c1, c2 = st.columns(2)
            with c1:
                max_alloc = st.slider(
                    "Max Security Allocation",
                    min_value=0.20, max_value=0.90, step=0.01,
                    value=float(params["max_security_alloc"]),
                    format="%.2f",
                    help="Max fraction of portfolio in a single security.",
                )
            with c2:
                max_risk = st.slider(
                    "Max Risk Per Trade",
                    min_value=0.005, max_value=0.050, step=0.001,
                    value=float(params["max_risk_per_trade"]),
                    format="%.3f",
                    help="Capital at risk per new buy as fraction of portfolio.",
                )

            st.markdown("**Weekly features — stop-loss construction**")
            c3, c4 = st.columns(2)
            with c3:
                atr_mult = st.slider(
                    "ATR Multiplier",
                    min_value=0.2, max_value=3.0, step=0.1,
                    value=float(params["atr_mult"]),
                    format="%.1f",
                    help="Multiplier for ATR component in weekly stop-loss.",
                    disabled=not raw_ok,
                )
            with c4:
                st_buffer = st.slider(
                    "Supertrend Buffer",
                    min_value=0.005, max_value=0.250, step=0.005,
                    value=float(params["st_buffer"]),
                    format="%.3f",
                    help="Percentage below Supertrend used as stop-loss floor.",
                    disabled=not raw_ok,
                )

            st.markdown("**Monthly features — buy price ceiling**")
            overshoot = st.slider(
                "Overshoot Coefficient",
                min_value=0.90, max_value=1.30, step=0.01,
                value=float(params["overshoot_coeff"]),
                format="%.2f",
                help="Max buy price = overshoot_coeff × prior-month swing peak.",
                disabled=not raw_ok,
            )

            st.markdown("**Buy signal filters**")
            c5, c6 = st.columns(2)
            with c5:
                crsi_thresh = st.slider(
                    "CRSI Threshold",
                    min_value=1.0, max_value=75.0, step=1.0,
                    value=float(params["crsi_threshold"]),
                    format="%.0f",
                    help="Entry only when CRSI < threshold (avoids overbought).",
                    disabled=not raw_ok,
                )
            with c6:
                risk_thresh = st.slider(
                    "Risk Threshold",
                    min_value=0.05, max_value=0.60, step=0.05,
                    value=float(params["risk_threshold"]),
                    format="%.2f",
                    help="Entry only when max risk % < threshold.",
                    disabled=not raw_ok,
                )

            st.markdown("---")
            label = (
                "Apply & Run  —  Full Pipeline + Backtest"
                if raw_ok
                else "Apply & Run  —  Backtest Only (engine params)"
            )
            submitted = st.form_submit_button(label, type="primary", use_container_width=True)

        if submitted:
            new_params = dict(
                max_security_alloc=max_alloc,
                max_risk_per_trade=max_risk,
                atr_mult=atr_mult,
                st_buffer=st_buffer,
                overshoot_coeff=overshoot,
                crsi_threshold=crsi_thresh,
                risk_threshold=risk_thresh,
            )
            st.session_state.params = new_params
            st.session_state.backtest_error = None

            if raw_ok:
                msg = "Running full engineering pipeline + backtest (30–60 s)…"
                runner = lambda: _run_full_pipeline(new_params)
            else:
                msg = "Running backtest with updated engine parameters…"
                runner = lambda: _run_backtest(fd, new_params)

            with st.spinner(msg):
                res = runner()

            if "error" in res:
                st.session_state.backtest_error = res["error"]
                st.error("Pipeline failed:")
                st.code(res["error"])
            else:
                st.session_state.backtest_results = res
                if "features_d" in res:
                    st.session_state.features_d = res["features_d"]
                st.success("Done. Switch to Backtest Results or Daily Signals to see updates.")
                st.rerun()


if __name__ == "__main__":
    main()
