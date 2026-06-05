import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

_EVENT_COLS = [
    "event_id", "trade_count", "ticker", "event_type", "date",
    "execution_price", "units_transacted", "units_held_after",
    "avg_buy_price", "stop_loss_price", "stop_loss_pct",
    "position_size_used", "portfolio_value", "cash_after",
    "realized_gain_abs", "realized_gain_pct",
]


class TradeLogger:
    """Accumulates trade events and serialises them to a DataFrame."""

    def __init__(self) -> None:
        self._events: List[Dict[str, Any]] = []
        self._event_id: int = 0

    def log_event(
        self,
        *,
        trade_count: int,
        ticker: str,
        event_type: str,
        event_date: Any,
        execution_price: float,
        units_transacted: int,
        units_held_after: int,
        avg_buy_price: float,
        stop_loss_price: float,
        stop_loss_pct: float,
        position_size_used: float,
        portfolio_value: float,
        cash_after: float,
        realized_gain_abs: float = float("nan"),
        realized_gain_pct: float = float("nan"),
    ) -> None:
        """Append one trade event to the in-memory log."""
        self._event_id += 1
        self._events.append(
            {
                "event_id": self._event_id,
                "trade_count": trade_count,
                "ticker": ticker,
                "event_type": event_type,
                "date": event_date,
                "execution_price": execution_price,
                "units_transacted": units_transacted,
                "units_held_after": units_held_after,
                "avg_buy_price": avg_buy_price,
                "stop_loss_price": stop_loss_price,
                "stop_loss_pct": stop_loss_pct,
                "position_size_used": position_size_used,
                "portfolio_value": portfolio_value,
                "cash_after": cash_after,
                "realized_gain_abs": realized_gain_abs,
                "realized_gain_pct": realized_gain_pct,
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return all logged events as a DataFrame with canonical column order."""
        if not self._events:
            return pd.DataFrame(columns=_EVENT_COLS)
        return pd.DataFrame(self._events)[_EVENT_COLS]
