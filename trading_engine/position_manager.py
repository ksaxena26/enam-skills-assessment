import logging
import math
from dataclasses import dataclass
from datetime import date
from typing import Optional, Tuple

from trading_engine.config import MAX_SECURITY_ALLOC, MAX_RISK_PER_TRADE

logger = logging.getLogger(__name__)


@dataclass
class PositionState:
    """Mutable per-ticker position state tracked through the backtest."""

    units_held: int = 0
    avg_buy_price: float = 0.0
    last_buy_price: float = 0.0
    allocated_value: float = 0.0
    trade_count: int = 0
    first_buy_date: Optional[date] = None


def compute_position_size(
    portfolio_value: float,
    current_close: float,
    stop_loss_price: float,
    units_held: int,
    ticker: str = "",
) -> Tuple[int, float, float]:
    """
    Compute (units_to_buy, raw_position_size, stop_loss_pct).

    Returns (0, 0.0, 0.0) when the trade must be skipped.
    The returned units_to_buy is already capped by allocation headroom but
    NOT yet capped by available cash — the caller applies the cash cap.
    """
    if stop_loss_price >= current_close:
        logger.warning(
            "%s: stop_loss_price %.4f >= current_close %.4f — skipping buy",
            ticker, stop_loss_price, current_close,
        )
        return 0, 0.0, 0.0

    stop_loss_pct = (current_close - stop_loss_price) / current_close

    if stop_loss_pct <= 0:
        logger.warning(
            "%s: stop_loss_pct <= 0 (%.6f) — skipping buy", ticker, stop_loss_pct
        )
        return 0, 0.0, 0.0

    raw_position_size = (MAX_RISK_PER_TRADE * portfolio_value) / stop_loss_pct

    available_headroom = (
        MAX_SECURITY_ALLOC * portfolio_value - units_held * current_close
    )
    if available_headroom <= 0:
        logger.info("%s: max allocation reached — no headroom available", ticker)
        return 0, raw_position_size, stop_loss_pct

    capped_size = min(raw_position_size, available_headroom)
    units_to_buy = math.floor(capped_size / current_close)

    return units_to_buy, raw_position_size, stop_loss_pct


def update_position_on_buy(
    state: PositionState,
    units_bought: int,
    execution_price: float,
    buy_date: date,
) -> None:
    """Update PositionState in-place after a successful buy execution."""
    new_units = state.units_held + units_bought
    if new_units > 0:
        state.avg_buy_price = (
            state.avg_buy_price * state.units_held
            + execution_price * units_bought
        ) / new_units
    state.units_held = new_units
    state.last_buy_price = execution_price
    state.allocated_value = state.units_held * execution_price
    state.trade_count += 1
    if state.first_buy_date is None:
        state.first_buy_date = buy_date


def reset_position(state: PositionState) -> None:
    """Reset per-ticker state after a full sell. Preserves trade_count."""
    state.units_held = 0
    state.avg_buy_price = 0.0
    state.last_buy_price = 0.0
    state.allocated_value = 0.0
    state.first_buy_date = None
