"""
Hyperparameter optimization for the trading engine using Optuna.

Objective : maximise CAGR (annualized_return_pct)
Sampler   : TPE (multivariate) with 20 random start-up trials
Trials    : 100 total
Storage   : SQLite (optuna_tuning.db) — dashboard connects here in real time

Tunable parameters
------------------
Engine (config.py)
  max_security_alloc   [0.20, 0.50]   (default 0.33)
  max_risk_per_trade   [0.005, 0.020] (default 0.010)

Weekly features — compute_features_w
  atr_mult             [0.50, 3.00]   (default 1.0)
  st_buffer            [0.005, 0.050] (default 0.02)

Monthly features — compute_features_m
  overshoot_coeff      [1.05, 1.30]   (default 1.15)

Buy flag — compute_buy_flag
  crsi_threshold       [30, 75]       (default 50)
  risk_threshold       [0.15, 0.60]   (default 0.40)

Usage
-----
    python trading_engine/optimize.py

Dashboard
---------
    http://127.0.0.1:8080   (launched automatically)
    pip install optuna-dashboard   # if not already installed
"""

import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

# ── Project root on path ───────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import optuna
import pandas as pd

# Silence engine/engineering INFO during trials; keep Optuna progress visible
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
optuna.logging.set_verbosity(optuna.logging.INFO)

from src.data_manager import load_ohlc_data
from src.engineering import (
    compute_buy_range_u,
    compute_features_d,
    compute_features_m,
    compute_features_w,
    compute_max_risk,
    get_batches,
    map_monthly_swings_to_daily,
    map_weekly_to_daily,
)
from src.utils.utils import daily_to_monthly_transform, daily_to_weekly_transform
from trading_engine.engine import run_backtest

# ── Study configuration ────────────────────────────────────────────────────────
STUDY_NAME       = "trading_engine_cagr_v2"
DB_PATH          = Path("optuna_tuning.db")
STORAGE          = f"sqlite:///{DB_PATH}"
N_STARTUP_TRIALS = 20    # random exploration before TPE activates
N_TRIALS         = 200   # total budget
DASHBOARD_PORT   = 8080


# ── Buy-flag helper with tunable thresholds ────────────────────────────────────
def _buy_flag(
    fd: pd.DataFrame,
    crsi_threshold: float,
    risk_threshold: float,
) -> pd.Series:
    """compute_buy_flag with configurable CRSI and risk thresholds."""
    conds = [
        fd["close"] > fd["supert_w"],
        fd["crsi"]  < crsi_threshold,
        fd["risk"]  < risk_threshold,
        fd["close"] < fd["max_buy_price_b1"],
    ]
    return pd.Series(np.where(np.all(conds, axis=0), 1, 0), index=fd.index)


# ── One-time precomputation ────────────────────────────────────────────────────
def precompute_base_features() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load OHLC and compute all features that contain NO tunable parameters.
    This runs once before the study starts; each trial only redoes the
    lightweight sl / max_buy_price arithmetic.
    """
    print("Loading OHLC data…")
    price_d = load_ohlc_data()
    price_w = daily_to_weekly_transform(price_d).drop(columns=["prev_close"])
    price_m = daily_to_monthly_transform(price_d).drop(columns=["prev_close"])
    price_d = price_d.drop(columns=["prev_close"])

    print("Computing base features (ATR, EMAs, RSI, Supertrend) — runs once…")
    features_d_base = pd.concat(
        [compute_features_d(b) for b in get_batches(price_d)],
        ignore_index=True,
    )
    features_w_base = pd.concat(
        [compute_features_w(b) for b in get_batches(price_w)],
        ignore_index=True,
    )
    features_m_base = pd.concat(
        [compute_features_m(b) for b in get_batches(price_m)],
        ignore_index=True,
    )
    print(
        f"Base features ready — {len(features_d_base):,} daily / "
        f"{len(features_w_base):,} weekly / "
        f"{len(features_m_base):,} monthly rows\n"
    )
    return features_d_base, features_w_base, features_m_base


# ── Objective factory ──────────────────────────────────────────────────────────
def make_objective(
    features_d_base: pd.DataFrame,
    features_w_base: pd.DataFrame,
    features_m_base: pd.DataFrame,
):
    """
    Return an Optuna objective that closes over the precomputed base features.
    The heavy indicators (Supertrend, ATR, RSI, EMA) are computed only once;
    each trial recomputes only the sl / max_buy_price columns and runs the
    backtest.
    """
    import trading_engine.config           as _cfg
    import trading_engine.engine           as _eng
    import trading_engine.position_manager as _pm

    def objective(trial: optuna.Trial) -> float:
        # ── Sample ────────────────────────────────────────────────────────────
        max_security_alloc = trial.suggest_float("max_security_alloc", 0.20,  0.90)
        max_risk_per_trade = trial.suggest_float("max_risk_per_trade", 0.005, 0.05)
        atr_mult           = trial.suggest_float("atr_mult",           0.20,  3.00)
        st_buffer          = trial.suggest_float("st_buffer",          0.005, 0.25)
        overshoot_coeff    = trial.suggest_float("overshoot_coeff",    0.9,  1.30)
        crsi_threshold     = trial.suggest_float("crsi_threshold",     1.0,  75.0)
        risk_threshold     = trial.suggest_float("risk_threshold",     0.05,  0.60)

        # ── Rebuild weekly stop-loss (atr_mult, st_buffer) ────────────────────
        # Only sl and sl_b1 change; all other weekly columns are reused as-is.
        features_w = features_w_base.copy()
        for _sym, idx in features_w.groupby("symbol").groups.items():
            grp = features_w.loc[idx]
            sl1 = grp[["sma30", "wdma30"]].min(axis=1) - atr_mult * grp["atr_14"]
            sl2 = (1 - st_buffer) * grp["supert"]
            sl3 = (1 - st_buffer) * grp["supert_b1"]
            sl  = pd.concat([sl1, sl2, sl3], axis=1).min(axis=1)
            features_w.loc[idx, "sl"]    = sl.values
            features_w.loc[idx, "sl_b1"] = sl.shift(1).values

        # ── Rebuild monthly max-buy-price (overshoot_coeff) ───────────────────
        features_m = features_m_base.copy()
        for _sym, idx in features_m.groupby("symbol").groups.items():
            grp = features_m.loc[idx]
            mbp = overshoot_coeff * grp["peak"]
            features_m.loc[idx, "max_buy_price"]    = mbp.values
            features_m.loc[idx, "max_buy_price_b1"] = mbp.shift(1).values

        # ── Assemble daily features ───────────────────────────────────────────
        fd = features_d_base.copy()
        fd = map_weekly_to_daily(fd, features_w, ["supert", "sl"])
        fd = map_monthly_swings_to_daily(fd, features_m)
        fd = compute_buy_range_u(fd, features_m)
        fd["risk"] = compute_max_risk(fd, "sl_w")
        fd["buy"]  = _buy_flag(fd, crsi_threshold, risk_threshold)
        fd = fd[fd["symbol"] != "NIFTYBEES"].copy()

        # ── Patch engine/position_manager module-level constants ──────────────
        # Single-threaded study (n_jobs=1) makes this safe.
        _eng.MAX_SECURITY_ALLOC = max_security_alloc
        _eng.MAX_RISK_PER_TRADE = max_risk_per_trade
        _pm.MAX_SECURITY_ALLOC  = max_security_alloc
        _pm.MAX_RISK_PER_TRADE  = max_risk_per_trade

        try:
            results = run_backtest(fd, write_outputs=False, run_analytics=False)
            stats   = results["summary_stats"]
            cagr    = float(stats.get("annualized_return_pct", math.nan))
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            return float("-inf")
        finally:
            # Always restore so interactive engine runs stay unaffected
            _eng.MAX_SECURITY_ALLOC = _cfg.MAX_SECURITY_ALLOC
            _eng.MAX_RISK_PER_TRADE = _cfg.MAX_RISK_PER_TRADE
            _pm.MAX_SECURITY_ALLOC  = _cfg.MAX_SECURITY_ALLOC
            _pm.MAX_RISK_PER_TRADE  = _cfg.MAX_RISK_PER_TRADE

        if not math.isfinite(cagr):
            return float("-inf")

        # Store secondary metrics for dashboard display
        trial.set_user_attr("n_trades",  int(stats.get("total_trades",        0)))
        trial.set_user_attr("sharpe",    float(stats.get("sharpe_ratio",      math.nan)))
        trial.set_user_attr("max_dd_pct",float(stats.get("max_drawdown_pct",  math.nan)))
        trial.set_user_attr("win_rate",  float(stats.get("win_rate",          math.nan)))
        return cagr

    return objective


# ── Dashboard launcher ─────────────────────────────────────────────────────────
def launch_dashboard() -> None:
    """Start optuna-dashboard in the background pointing at the SQLite DB."""
    import shutil

    # Locate the binary inside the active Python env's Scripts / bin directory
    binary = shutil.which("optuna-dashboard") or shutil.which("optuna-dashboard.EXE")

    if binary is None:
        print("optuna-dashboard not found. Install with: pip install optuna-dashboard")
        print(f"  Then run: optuna-dashboard {STORAGE} --port {DASHBOARD_PORT}\n")
        return

    try:
        proc = subprocess.Popen(
            [binary, STORAGE, "--host", "127.0.0.1", "--port", str(DASHBOARD_PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        # Give it a moment to bind the port, then check it started cleanly
        import time
        time.sleep(2)
        if proc.poll() is not None:          # process already exited
            err = proc.stderr.read().decode(errors="replace").strip()
            print(f"optuna-dashboard failed to start: {err}")
            print(f"  Run manually: optuna-dashboard {STORAGE} --port {DASHBOARD_PORT}\n")
        else:
            print(f"optuna-dashboard started  (PID {proc.pid})")
            print(f"  Open in browser:  http://127.0.0.1:{DASHBOARD_PORT}\n")
    except Exception as exc:
        print(f"Could not start optuna-dashboard: {exc}")
        print(f"  Run manually: optuna-dashboard {STORAGE} --port {DASHBOARD_PORT}\n")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Precompute heavy features once
    features_d_base, features_w_base, features_m_base = precompute_base_features()

    # 2. Launch dashboard before study starts (connects as soon as DB is created)
    launch_dashboard()

    # 3. Create or resume study
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=N_STARTUP_TRIALS,
        seed=42,
        multivariate=True,            # models inter-parameter correlations
        warn_independent_sampling=False,
    )
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,          # resume if DB already exists
    )

    print(
        f"Study: '{STUDY_NAME}'  |  direction: maximize CAGR\n"
        f"Budget: {N_TRIALS} trials  "
        f"({N_STARTUP_TRIALS} random start-up + {N_TRIALS - N_STARTUP_TRIALS} TPE)\n"
    )

    objective = make_objective(features_d_base, features_w_base, features_m_base)
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    # 4. Print results
    best = study.best_trial
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"  Best CAGR   : {best.value:.4f}%")
    print(f"  Trial #     : {best.number}")
    print("\n  Best parameters:")
    for k, v in sorted(best.params.items()):
        print(f"    {k:<26} {v:.6f}")
    print("\n  Secondary metrics (best trial):")
    for k, v in sorted(best.user_attrs.items()):
        if k != "error":
            print(f"    {k:<26} {v}")
    print(f"\n  Dashboard: http://127.0.0.1:{DASHBOARD_PORT}")
    print(f"  DB path  : {DB_PATH.resolve()}")
