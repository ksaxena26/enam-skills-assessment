from ta.trend import sma_indicator, ema_indicator
from ta.volatility import average_true_range
from ta.momentum import rsi
import pandas_ta as pta
import pandas as pd
import numpy as np
from src.utils.utils import reposition_column
from scipy.stats import percentileofscore


# ── Multi-timeframe helpers ────────────────────────────────────────────────────
_WEEKLY_FREQ = 'W-FRI'
_MONTHLY_FREQ = 'ME'

# Daily columns that gain _w / _m counterparts → renamed to _d in the full pipeline
_DAILY_RENAMES = {
    'atr_14': 'atr_14_d',
    'net_change_atr': 'net_change_atr_d',
    'rsi_7': 'rsi_7_d',
    'rsi_7_sma3': 'rsi_7_sma3_d',
    'supert': 'supert_d',
    'supertd': 'supertd_d',
}


def calculate_percentile(series: np.ndarray) -> float:
    return percentileofscore(series, series[-1], kind='strict')


def rolling_percentile(price_change: pd.Series, window: int) -> pd.Series:
    return price_change.rolling(window).apply(calculate_percentile, raw=True)


def calc_streak(price_change: pd.Series) -> np.ndarray:
    up = price_change > 0
    down = price_change < 0

    up_streak = (
        up.groupby((~up).cumsum())
        .cumsum()
        .where(up, 0)
    )
    down_streak = (
        down.groupby((~down).cumsum())
        .cumsum()
        .where(down, 0)
    )

    return (up_streak - down_streak).to_numpy()


def compute_atr(group: pd.DataFrame, window: int = 14) -> pd.Series:
    if len(group) < window:
        return pd.Series(np.nan, index=group.index)
    return average_true_range(group['high'], group['low'], group['close'], window=window)


def rolling_linreg_1d(y: np.ndarray, window: int, return_slope: bool = False) -> np.ndarray:
    n = len(y)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    k_y = np.ones(window)
    k_xy = np.arange(window, dtype=float)[::-1]

    # rolling sum of y and rolling sum of x*y (using reversed x weights)
    sum_y = np.convolve(y, k_y, mode='full')[window - 1:n]
    sum_xy = np.convolve(y, k_xy, mode='full')[window - 1:n]

    y_mean = sum_y / window
    slope = (sum_xy - window * x_mean * y_mean) / x_var
    intercept = y_mean - slope * x_mean

    nans = np.full(window - 1, np.nan)

    if return_slope:
        return np.concatenate([nans, slope])

    y_hat_last = intercept + slope * (window - 1)
    return np.concatenate([nans, y_hat_last])


def sharpe_momentum(sharpe: np.ndarray, freq: str, window: int) -> np.ndarray:
    freq_map = {'d': (252, 10), 'w': (52, 5), 'm': (12, 2)}
    n, factor = freq_map[freq]

    lr_line = rolling_linreg_1d(sharpe, n)

    total = len(lr_line)
    roc = np.concatenate([
        np.full(window - 1, np.nan),
        lr_line[window - 1:] - lr_line[:total - window + 1]
    ])

    return factor * roc


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                suffix: str = None) -> pd.DataFrame:
    """Return 2-col DataFrame with supert/supertd (or supert_{suffix}/supertd_{suffix})."""
    col_s = 'supert' if suffix is None else f'supert_{suffix}'
    col_d = 'supertd' if suffix is None else f'supertd_{suffix}'
    try:
        st = pta.supertrend(high, low, close, length=10, multiplier=3)
        if st is None or st.empty:
            raise ValueError
        cols = st.iloc[:, :2].copy()
        cols.columns = [col_s, col_d]
        return cols
    except Exception:
        return pd.DataFrame({col_s: np.nan, col_d: np.nan}, index=high.index)


def compute_features_d(price_df: pd.DataFrame) -> pd.DataFrame:
    data = price_df.copy()
    close = data['close']
    high = data['high']
    low = data['low']
    open_ = data['open']
    # delpct = data['delpct']

    # 1. Lagged OHLC b1–b6
    for i in range(1, 7):
        data[f'open_b{i}'] = open_.shift(i)
        data[f'high_b{i}'] = high.shift(i)
        data[f'low_b{i}'] = low.shift(i)
        data[f'close_b{i}'] = close.shift(i)

    # 2. Price changes
    data['price_change'] = close.pct_change()
    data['price_change_across_b50'] = close.pct_change(50)
    data['price_change_across_b252'] = close.pct_change(252)

    # 3. Sharpe 252
    def _rolling_sharpe(s, window=252):
        mu = s.rolling(window).mean()
        sigma = s.rolling(window).std()
        sharpe = mu / sigma * np.sqrt(window)
        sharpe[sigma == 0] = np.nan
        return sharpe

    data['sharpe_252'] = _rolling_sharpe(data['price_change'])
    data['sharpe_252_sma3'] = sma_indicator(data['sharpe_252'], window=3)

    # 4. Sharpe momentum
    data['sharpe_mmt'] = sharpe_momentum(
        data['sharpe_252'].to_numpy(dtype=float), freq='d', window=10
    )

    # 5. Supertrend
    st_df = _supertrend(high, low, close)
    data['supert'] = st_df['supert'].values
    data['supertd'] = st_df['supertd'].values

    # 6. Rolling highs
    data['52w_high'] = close.rolling(252, min_periods=2).max()
    data['2yr_high'] = close.rolling(504, min_periods=2).max()
    data['3yr_high'] = close.rolling(756, min_periods=2).max()

    # 7. EMAs, SMAs, delivery %
    data['ema10'] = ema_indicator(close, window=10)
    data['ema21'] = ema_indicator(close, window=21)
    data['sma50'] = sma_indicator(close, window=50)
    data['sma100'] = sma_indicator(close, window=100)
    data['sma200'] = sma_indicator(close, window=200)
    # data['delpct_sma10'] = sma_indicator(delpct, window=10)
    # data['delpct_sma50'] = sma_indicator(delpct, window=50)
    # data['delpct_sma100'] = sma_indicator(delpct, window=100)
    # data['delpct_sma200'] = sma_indicator(delpct, window=200)

    # 8. ATR and net change in ATR units
    data['atr_14'] = compute_atr(data, 14)
    net_change = (close - data['close_b1']) / data['atr_14']
    net_change = net_change.replace([np.inf, -np.inf], np.nan).round(4)
    data['net_change_atr'] = net_change

    # 9. RSI variants
    data['rsi_7'] = rsi(close, window=7)
    data['rsi_7_sma3'] = sma_indicator(data['rsi_7'], window=3)
    data['rsi_14'] = rsi(close, window=14)
    data['rsi_14_sma3'] = sma_indicator(data['rsi_14'], window=3)

    # 10. ConnorsRSI
    streak_series = pd.Series(calc_streak(data['price_change']), index=data.index)
    data['crsi'] = (
        rsi(close, window=3)
        + rsi(streak_series, window=2)
        + rolling_percentile(data['price_change'], 100)
    ) / 3

    return data


# ── Standalone weekly / monthly feature functions ─────────────────────────────

def wilder_ma(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(alpha=1 / window, adjust=False).mean()


def calc_drawdown_streak(
        price_change, price_change_b1, price_change_b2,
        high, high_b1, high_b2,
        price_struct, price_struct_b1, price_struct_b2
) -> np.ndarray:
    conditions = [price_change < 0,  # Condition 1
                  (price_change >= 0) &  # Condition 2
                  (price_struct == "LHLL"),
                  (price_change >= 0) &  # Condition 3
                  ((price_struct == "LHHL") |
                   (price_struct == "HHLL")) &
                  (high < np.maximum(high_b1, high_b2)) &
                  ((price_change_b1 < 0) |
                   (price_struct_b1 == 'LHLL')),
                  (price_change >= 0) &  # Condition 4
                  (high < np.maximum(high_b1, high_b2)) &
                  (price_change_b2 < 0) &
                  (price_struct_b2 == 'LHLL'), ]
    options = [1, 1, 1, 1,]
    drawdown = np.select(conditions, options, 0)

    return calc_streak(pd.Series(drawdown))



def compute_features_w(price_df: pd.DataFrame) -> pd.DataFrame:
    data = price_df.copy()
    close = data['close']

    # 1. Lagged OHLC b1..6
    for col in ['open', 'high', 'low', 'close']:
        for i in range(1, 7):
            data[f'{col}_b{i}'] = data[col].shift(i)

    # 2. Price change
    data['price_change'] = close.pct_change()

    # 3. Moving averages and lags b1..6
    data['ema10'] = ema_indicator(close, window=10)
    data['sma30'] = sma_indicator(close, window=30)
    data['wdma30'] = wilder_ma(close, 30)
    for col in ['ema10', 'sma30', 'wdma30']:
        for i in range(1, 7):
            data[f'{col}_b{i}'] = data[col].shift(i)

    # 4. Sharpe(104, sqrt(52)) and momentum
    mu = data['price_change'].rolling(104).mean()
    sig = data['price_change'].rolling(104).std()
    data['sharpe_104'] = (
        np.divide(mu, sig, out=np.full_like(mu, np.nan, dtype=float), where=sig != 0) * np.sqrt(52)
    )
    data['sharpe_mmt'] = sharpe_momentum(data['sharpe_104'].to_numpy(dtype=float), freq='d', window=10)

    # 5. Supertrend and lags b1..4
    data = pd.concat([data, _supertrend(data['high'], data['low'], data['close'])], axis=1)
    for i in range(1, 5):
        data[f'supert_b{i}'] = data['supert'].shift(i)
    for i in range(1, 5):
        data[f'supertd_b{i}'] = data['supertd'].shift(i)

    # 6. RSI(7), smoothed RSI, lags b1..2
    data['rsi_7'] = rsi(close, window=7)
    data['rsi_7_sma3'] = sma_indicator(data['rsi_7'], window=3)
    for i in range(1, 3):
        data[f'rsi_7_sma3_b{i}'] = data['rsi_7_sma3'].shift(i)

    # 7. ATR(14), net change in ATR units, lags b1..4
    data['atr_14'] = compute_atr(data, 14)
    data['net_change_atr'] = (
        (close - data['close_b1']) / data['atr_14']
    ).replace([np.inf, -np.inf], np.nan).round(4)
    for i in range(1, 5):
        data[f'net_change_atr_b{i}'] = data['net_change_atr'].shift(i)

    # 8. ConnorsRSI and lags b1..2
    streak_series = pd.Series(calc_streak(data['price_change']), index=data.index)
    data['crsi'] = (
        rsi(close, window=3)
        + rsi(streak_series, window=2)
        + rolling_percentile(data['price_change'], 100)
    ) / 3
    for i in range(1, 3):
        data[f'crsi_b{i}'] = data['crsi'].shift(i)

    data = reposition_column(data, 'symbol', 1)

    return data


def compute_features_m(price_df: pd.DataFrame) -> pd.DataFrame:
    data = price_df.copy()
    close = data['close']

    # 1. Lagged OHLC b1..2 — single concat
    data = pd.concat([data, pd.DataFrame(
        {f'{col}_b{i}': data[col].shift(i)
         for col in ['open', 'high', 'low', 'close']
         for i in range(1, 3)},
        index=data.index,
    )], axis=1)

    # 2. Rolling 24m / 36m highs and b1 lags
    data['high_24m'] = close.rolling(24, min_periods=12).max()
    data['high_36m'] = close.rolling(36, min_periods=12).max()
    data['high_24m_b1'] = data['high_24m'].shift(1)
    data['high_36m_b1'] = data['high_36m'].shift(1)

    # 3. Price changes, lags, backward cross-period, forward cross-period
    data['price_change'] = close.pct_change()
    data = pd.concat([data, pd.DataFrame(
        {f'price_change_b{i}': data['price_change'].shift(i) for i in range(1, 3)},
        index=data.index,
    )], axis=1)
    data['price_change_across_b6'] = close.pct_change(6)
    data['price_change_across_b12'] = close.pct_change(12)
    data = pd.concat([data, pd.DataFrame(
        {f'price_change_across_f{i}': close.pct_change(i).shift(-i) for i in [1, 2, 3, 6]},
        index=data.index,
    )], axis=1)

    # 4. Sharpe (12/24/36-period, sqrt(12) annualisation) and momentum
    def _sharpe(w):
        mu = data['price_change'].rolling(w).mean()
        sig = data['price_change'].rolling(w).std()
        return np.divide(mu, sig, out=np.full_like(mu, np.nan, dtype=float), where=sig != 0) * np.sqrt(12)

    for w in [12, 24, 36]:
        data[f'sharpe_{w}'] = _sharpe(w)
    data['sharpe_mmt'] = sharpe_momentum(data['sharpe_24'].to_numpy(dtype=float), freq='d', window=10)

    # 5. Supertrend
    data = pd.concat([data, _supertrend(data['high'], data['low'], data['close'])], axis=1)

    # 6. EMA(5) / EMA(10) and lags b1..2
    data['ema5'] = ema_indicator(close, window=5)
    data['ema10'] = ema_indicator(close, window=10)
    for col in ['ema5', 'ema10']:
        data = pd.concat([data, pd.DataFrame(
            {f'{col}_b{i}': data[col].shift(i) for i in range(1, 3)},
            index=data.index,
        )], axis=1)

    # 7. ATR(14), net_change_atr, lags b1..4
    data['atr_14'] = compute_atr(data, 14)
    data['net_change_atr'] = (
        (close - data['close_b1']) / data['atr_14']
    ).replace([np.inf, -np.inf], np.nan).round(4)
    data = pd.concat([data, pd.DataFrame(
        {f'net_change_atr_b{i}': data['net_change_atr'].shift(i) for i in range(1, 5)},
        index=data.index,
    )], axis=1)

    # 8. NR4 — narrow-range flag: current or prior 4-period window with max |net_change_atr| < 0.35
    nca_curr = pd.concat([data['net_change_atr'], data['net_change_atr_b1'],
                           data['net_change_atr_b2'], data['net_change_atr_b3']], axis=1)
    nca_prev = pd.concat([data['net_change_atr_b1'], data['net_change_atr_b2'],
                           data['net_change_atr_b3'], data['net_change_atr_b4']], axis=1)
    data['nr4'] = (nca_curr.abs().max(axis=1) < 0.35) | (nca_prev.abs().max(axis=1) < 0.35)

    # 9. RSI(7), smoothed RSI, lags b1..2
    data['rsi_7'] = rsi(close, window=7)
    data['rsi_7_sma3'] = sma_indicator(data['rsi_7'], window=3)
    data = pd.concat([data, pd.DataFrame(
        {f'rsi_7_sma3_b{i}': data['rsi_7_sma3'].shift(i) for i in range(1, 3)},
        index=data.index,
    )], axis=1)

    # 10. ConnorsRSI and lags b1..2
    streak_series = pd.Series(calc_streak(data['price_change']), index=data.index)
    data['crsi'] = (
        rsi(close, window=3)
        + rsi(streak_series, window=2)
        + rolling_percentile(data['price_change'], 100)
    ) / 3
    data = pd.concat([data, pd.DataFrame(
        {f'crsi_b{i}': data['crsi'].shift(i) for i in range(1, 3)},
        index=data.index,
    )], axis=1)

    # 11. Price structure: HH/LH × HL/LL classification, lags b1..5
    h, l, h1, l1 = data['high'], data['low'], data['high_b1'], data['low_b1']
    data['price_struct'] = np.select(
        [(h > h1) & (l > l1),  (h > h1) & (l <= l1),
         (h <= h1) & (l > l1), (h <= h1) & (l <= l1)],
        ['HHHL', 'HHLL', 'LHHL', 'LHLL'],
        default=np.nan,
    )
    data = pd.concat([data, pd.DataFrame(
        {f'price_struct_b{i}': data['price_struct'].shift(i) for i in range(1, 6)},
        index=data.index,
    )], axis=1)

    # 12. Drawdown streak (stub — full logic TBD)
    data['drawdown_streak'] = calc_drawdown_streak(
        data['price_change'],   data['price_change_b1'], data['price_change_b2'],
        data['high'],           data['high_b1'],          data['high_b2'],
        data['price_struct'],   data['price_struct_b1'],  data['price_struct_b2'],
    )

    # 13. Rally month flag, lags b1..3, forwards f1..3, and derived label
    data['rally_month'] = (
        (close >= data['high_24m_b1']) & (data['price_change'] > 0)
    ).astype(int)
    data = pd.concat([data, pd.DataFrame(
        {**{f'rally_month_b{i}': data['rally_month'].shift(i)  for i in range(1, 4)},
         **{f'rally_month_f{i}': data['rally_month'].shift(-i) for i in range(1, 4)}},
        index=data.index,
    )], axis=1)
    data['rally_in_next_3_prds'] = (
        (data['rally_month_f1'] + data['rally_month_f2'] + data['rally_month_f3']) > 0
    ).astype(int)
    data['label'] = (
        (data['rally_in_next_3_prds'] == 1) | (data['price_change_across_f3'] >= 0.2)
    ).astype(int)

    data = reposition_column(data, 'symbol', 1)

    return data.copy()
